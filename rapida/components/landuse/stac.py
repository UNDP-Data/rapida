import json
import logging
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime
from queue import Queue
from typing import Optional
from calendar import monthrange
from osgeo import gdal
from datetime import date
from rich.progress import Progress
import pystac_client
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point, MultiPoint, Polygon
import time

from rapida.components.landuse.sentinel_item import SentinelItem

logger = logging.getLogger('rapida')


def create_date_range(target_year: int, target_month: Optional[int] = None, duration: int = 6) -> str:
    """
    Generate a date range string in the format 'YYYY-MM-DD/YYYY-MM-DD'.

    The end date is determined based on the provided target year and optional target month:
    - If target_year is None or in the future → use current year.
    - If target_month is None:
        - If target_year is current year → use today as end_date.
        - If target_year is past → use December as default month.
    - If target_month is in the future (within current year) → clamp to current month.
    - end_date = today (if current year and current/future month), else last day of target_month.
    - start_date = end_date - `duration` months (manual calculation, no external lib).

    The start date is computed by subtracting `duration` months from the end date,
    accounting for varying month lengths and year boundaries.

    :param target_year: The year of interest.
    :param target_month: The month of interest (1–12), optional.
    :param duration: Number of months to go back from the end date (default is 6).
    :return: A date range string in the format 'YYYY-MM-DD/YYYY-MM-DD'.
    """
    today = date.today()
    current_year = today.year
    current_month = today.month

    def subtract_months(from_date: date, months: int) -> date:
        year = from_date.year
        month = from_date.month - months

        while month <= 0:
            month += 12
            year -= 1

        day = min(from_date.day, monthrange(year, month)[1])
        return date(year, month, day)

    # normalize year
    if target_year is None or target_year > current_year:
        target_year = current_year

    # normalize month
    if target_month is None:
        if target_year < current_year:
            target_month = 12  # Use December for past years
    else:
        if target_year == current_year and target_month > current_month:
            target_month = current_month

            # Determine end_date
    if target_year == current_year:
        if target_month is None or target_month >= current_month:
            end_date = today
        else:
            last_day = monthrange(target_year, target_month)[1]
            end_date = date(target_year, target_month, last_day)
    else:
        if target_month is None:
            end_date = date(target_year, 12, 31)
        else:
            last_day = monthrange(target_year, target_month)[1]
            end_date = date(target_year, target_month, last_day)

    start_date = subtract_months(end_date, duration)

    return f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"


def merge_or_voronoi(df: gpd.GeoDataFrame, scene_size=110000) -> list[Polygon]:
    """
    Merge or split input polygons into voronoi depending on their spatial extent.

    If the total area covered by the input polygons is smaller than or equal to a Sentinel-2 scene
    (approximately 110 km x 110 km), merge all polygons into a single one for STAC search.

    If the area is larger, split it using Voronoi polygons by creating multiple points inside a merged polygon.

    :param df: Input GeoDataFrame containing search polygons.
    :param scene_size:  Approximate size of a Sentinel-2 scene in meters (default is 110000).
    :return A list of merged or split polygons to be used for STAC search.
    """
    original_crs = df.crs
    # reproject to 3857 to ease to compute area in meters.
    df_3857 = df.to_crs(epsg=3857)

    merged = df_3857.geometry.union_all()
    merged = merged.simplify(tolerance=1000, preserve_topology=False)
    total_area = merged.area

    if total_area <= scene_size ** 2:
        # Area is small enough to be covered by a single STAC search
        return [gpd.GeoSeries([merged], crs=3857).to_crs(original_crs).iloc[0]]

    # Generate internal grid points spaced equally based on scene size
    minx, miny, maxx, maxy = merged.bounds
    # create double size of scene to be able to cover the whole scene area
    spacing = scene_size * 2
    nx = int(np.ceil((maxx - minx) / spacing))
    ny = int(np.ceil((maxy - miny) / spacing))

    points = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            x = minx + i * spacing
            y = miny + j * spacing
            pt = Point(x, y)
            if merged.contains(pt):
                points.append(pt)

    if len(points) < 2:
        return [gpd.GeoSeries([merged], crs=3857).to_crs(original_crs).iloc[0]]

    multipoint = MultiPoint(points)
    voronoi_geom = shapely.voronoi_polygons(multipoint, extend_to=merged, only_edges=False)

    # clip voronoi polygons by original polygon
    polygons_3857 = [
        poly.intersection(merged)
        for poly in voronoi_geom.geoms
        if poly.is_valid and not poly.is_empty
    ]

    # reproject final outputs to original projection
    return gpd.GeoSeries(polygons_3857, crs=3857).to_crs(original_crs).tolist()

def search_stac_items(stac_client,
                      collection_id: str,
                      df_polygon: gpd.GeoDataFrame,
                      datetime_range: str,
                      target_assets: dict[str, str],
                      max_workers: int = 5,
                      progress: Progress = None,
                      ):
    """
    Search stac items for covering project geodataframe area

    :param stac_client: stac client
    :param collection_id: collection id
    :param df_polygon: Polygon dataframe for searching
    :param target_assets: target assets
    :param max_workers: maximum number of workers
    :param progress: rich progress object
    """

    search_task = None
    if progress:
        search_task = progress.add_task(f"[green]Searching STAC Items for {datetime_range}", total=len(df_polygon))

    latest_per_tile = {}
    lock = threading.Lock()

    def search_single_polygon(single_geom):
        nonlocal latest_per_tile

        fc_geojson_str = gpd.GeoDataFrame(geometry=[single_geom], crs=df_polygon.crs).to_json()
        fc_geojson = json.loads(fc_geojson_str)
        first_feature = fc_geojson["features"][0]

        search = stac_client.search(
            collections=[collection_id],
            intersects=first_feature,
            query={"eo:cloud_cover": {"lt": 5}},
            datetime=datetime_range,
        )
        items = list(search.items())

        with lock:
            for item in items:
                tile_id = item.properties.get("grid:code")
                if tile_id is None:
                    continue

                cloud_cover = item.properties.get("eo:cloud_cover")
                if cloud_cover is None:
                    continue

                # if item does not have all required assets, skip it.
                if not all(asset in item.assets for asset in target_assets):
                    continue
                existing = latest_per_tile.get(tile_id)
                if existing is None:
                    latest_per_tile[tile_id] = item
                else:
                    existing_cloud = existing.properties.get("eo:cloud_cover")
                    if existing_cloud is None:
                        latest_per_tile[tile_id] = item
                    elif cloud_cover < existing_cloud:
                        # Update item if cloud cover is lower even it is older item
                        latest_per_tile[tile_id] = item
                    elif item.datetime > existing.datetime:
                        # Otherwise, newer image is used
                        latest_per_tile[tile_id] = item

        if progress and search_task:
            progress.update(search_task, advance=1)

    work_geoms = merge_or_voronoi(df_polygon)
    # gdf_out = gpd.GeoDataFrame(geometry=work_geoms, crs=df_polygon.crs)
    # gdf_out.to_file("voronoi_regions.geojson", driver="GeoJSON")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(search_single_polygon, geom) for geom in work_geoms]
        concurrent.futures.wait(futures)

    if progress and search_task:
        progress.remove_task(search_task)

    return latest_per_tile


def make_buffer_polygon(geopackage_file_path, polygons_layer_name, buffer=1000, gpkg_name="cutline.gpkg"):
    """
    load clip layer to make 1km buffer
    prediction model does not work along edge of data, so 1km buffer can be created before clipping.

    also, make simplify with tolerance of 1km to reduce nodes.

    In some circumstances, I got error like "TopologyException: side location conflict",
    so use make_valid to fix some invalid polygons

    :param: geopackage_file_path: path to geopackage file
    :param: polygons_layer_name: layer name
    :param: buffer: buffer size
    :param: gpkg_name: name of gpkg file
    """
    df_polygon = gpd.read_file(geopackage_file_path, layer=polygons_layer_name)
    original_crs = df_polygon.crs

    merged = df_polygon.geometry.unary_union
    buffered = merged.buffer(buffer)
    simplified = buffered.simplify(tolerance=buffer, preserve_topology=True)
    simplified_geom = gpd.GeoDataFrame(geometry=[simplified], crs=original_crs)

    simplified_geom["geometry"] = simplified_geom["geometry"].make_valid()

    tmp_cutline_path = os.path.join(os.path.dirname(geopackage_file_path), gpkg_name)
    simplified_geom.to_file(tmp_cutline_path, layer=polygons_layer_name, driver="GPKG")
    return tmp_cutline_path


async def download_stac(
    stac_url: str,
    collection_id: str,
    geopackage_file_path: str,
    polygons_layer_name: str,
    output_file: str,
    target_year: int,
    target_month:int,
    target_assets: dict[str, str],
    target_srs,
    progress: Progress = None,
    max_workers: int = 2,
):
    """
    download STAC data from Earth Search to create tiff file for each asset (eg, B02, B03) required

    :param stac_url: STAC root URL
    :param collection_id: collection id
    :param geopackage_file_path: path to geopackage file
    :param polygons_layer_name: name of layer polygon layer in geopackage to mask
    :param output_file: output file path
    :param target_year: target year
    :param target_assets: target assets.
    :param target_srs: target projection CRS
    :param progress: rich progress object
    :param max_workers: maximum number of workers to download JP2 file concurrently
    :return the list of output files
    """
    t1 = time.time()

    stac_task = None
    if progress:
        stac_task = progress.add_task(f"[cyan]Connecting to {stac_url}/{collection_id}", total=None)

    client = pystac_client.Client.open(stac_url)

    df_polygon = gpd.read_file(geopackage_file_path, layer=polygons_layer_name)
    df_polygon.to_crs(epsg=4326, inplace=True)

    if progress and stac_task:
        progress.update(stac_task, description="[yellow]Searching for STAC items...")

    datetime_range = create_date_range(target_year, target_month, duration=12)
    logger.debug(f"datetime range for searching: {datetime_range}")
    latest_per_tile = search_stac_items(stac_client=client,
                      collection_id=collection_id,
                      df_polygon=df_polygon,
                      datetime_range=datetime_range,
                      target_assets=target_assets,
                      progress=progress,)

    tmp_cutline_path = make_buffer_polygon(geopackage_file_path, polygons_layer_name)

    sentinel_items =  []
    for item in latest_per_tile.values():
        sentinel_items.append(SentinelItem(item, mask_file=tmp_cutline_path, mask_layer=polygons_layer_name))

    if progress and stac_task:
        progress.update(stac_task, description=f"[cyan]Preparing to download {len(sentinel_items)} items", total=len(sentinel_items))

    t2 = time.time()
    logger.debug(f"STAC Item search: {t2 - t1} seconds")

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    predict_task = None
    predict_task_lock = threading.Lock()

    predict_queue = Queue()
    completed_items = []

    def downloader(item: SentinelItem):
        try:
            item.download_assets(download_dir=output_dir, progress=progress)
            if not os.path.exists(item.predicted_file):
                # if predicted file does not exist in the folder, add prediction task to the queue
                predict_queue.put(item)
            if progress and stac_task:
                progress.update(stac_task, advance=1)
        except Exception as e:
            logger.error(f"Download failed for {item.item.id}: {e}")

    def predictor():
        nonlocal predict_task
        while True:
            item = predict_queue.get()
            if item is None:
                break
            with predict_task_lock:
                if predict_task is None and progress:
                    predict_task = progress.add_task("[cyan]Predicting...", total=len(sentinel_items))

            progress.update(predict_task, description=f"[cyan]Predicting {item.item.id}")
            item.predict(progress=progress)
            completed_items.append(item)
            progress.update(predict_task, advance=1)
            predict_queue.task_done()

    prediction_thread = threading.Thread(target=predictor)
    prediction_thread.start()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in sentinel_items:
            executor.submit(downloader, item)

    if progress and stac_task:
        progress.update(stac_task, description=f"[cyan]Downloaded {len(sentinel_items)} items", total=None)

    predict_queue.join()
    predict_queue.put(None)
    prediction_thread.join()

    if progress and predict_task:
        progress.remove_task(predict_task)

    if os.path.exists(tmp_cutline_path):
        os.remove(tmp_cutline_path)

    t3 = time.time()
    logger.debug(f"STAC Items processed: {t3 - t2} seconds")

    if progress and stac_task:
        progress.update(stac_task, description="[green]Creating mosaic...")

    # sort items by datetime and cloud cover
    sentinel_items_sorted = sorted(
        sentinel_items,
        key=lambda item: (
            item.item.datetime or datetime.min,
            item.item.properties.get("eo:cloud_cover", float("inf"))
        )
    )

    # for debug en sure items are sorted by datetime and cloud cover
    for item in sentinel_items_sorted:
        dt = item.item.datetime.isoformat() if item.item.datetime else "None"
        cc = item.item.properties.get("eo:cloud_cover", "None")
        logger.debug(f"{item.id}: datetime = {dt}, cloud_cover = {cc}, file = {item.predicted_file}")

    prediction_files = [item.predicted_file for item in sentinel_items_sorted]

    gdal.BuildVRT(output_file, prediction_files,
                  resampleAlg="nearest",
                  outputSRS=target_srs,
                  VRTNodata=255,
                  addAlpha=False)

    if progress and stac_task:
        progress.update(stac_task, description="[green]Created mosaic")
        progress.remove_task(stac_task)

    t4 = time.time()
    logger.debug(f"Download completed: {t4 - t1} seconds")

    return output_file

