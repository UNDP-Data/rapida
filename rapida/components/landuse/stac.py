import asyncio
import json
import logging
import os
from glob import glob
from collections import defaultdict
import concurrent.futures
import threading
from typing import Optional
from calendar import monthrange
from osgeo import gdal
from datetime import date
from rich.progress import Progress
import pystac_client
import rasterio
import httpx
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point, MultiPoint, Polygon
import time

from rapida.components.landuse.sentinel_item import SentinelItem

logger = logging.getLogger('rapida')


def interpolate_stac_source(source: str) -> dict[str, str]:
    """
    Interpolate stac source. Source of stac should be defined like below:

    {stac_id}:{collection_id}:{target band value}

    :param source: stac source
    :return: dist consist of id, collection and value
    """
    parts = source.split(':')
    assert len(parts) == 3, 'Invalid source definition'
    stac_id, collection, target_value = parts
    return {
        'id': stac_id,
        'collection': collection,
        'value': target_value
    }


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


def get_bounds_and_resolution(file_paths):
    bounds = []
    xres_list = []
    yres_list = []

    for fp in file_paths:
        with rasterio.open(fp) as src:
            bounds.append(src.bounds)
            xres, yres = src.res
            xres_list.append(xres)
            yres_list.append(yres)

    left = min(b.left for b in bounds)
    bottom = min(b.bottom for b in bounds)
    right = max(b.right for b in bounds)
    top = max(b.top for b in bounds)

    xRes = min(xres_list)
    yRes = min(yres_list)

    return (left, bottom, right, top), xRes, yRes


def crop_asset_files(base_dir,
                     target_srs,
                     asset_nodata: dict[str, int],
                     progress: Progress = None,):
    """
    Create a VRT for each asset (eg, B02, B03), then crop downloaded data by project area.
    The output file is geotiff file.

    :param base_dir: Root directory for downloaded sentinel data
    :param target_srs: target projection CRS
    :param progress: rich progress object
    """
    post_task = None
    if progress is not None:
        post_task = progress.add_task(
            description=f'[red] Postprocessing downloaded data', total=None)

    band_files = defaultdict(list)

    # delete all vrt and tif exists under base_dir
    for fname in os.listdir(base_dir):
        if fname.endswith((".tif", ".vrt", ".xml")):
            path = os.path.join(base_dir, fname)
            if os.path.isfile(path):
                os.remove(path)

    for jp2_path in glob(os.path.join(base_dir, "**", "B??.tif"), recursive=True):
        filename = os.path.basename(jp2_path)
        band_name = os.path.splitext(filename)[0]
        band_files[band_name].append(jp2_path)


    output_files = []

    if progress is not None and post_task is not None:
        progress.update(post_task, description=f'[red] Postprocessing...', total=len(band_files))

    for band_name, files in band_files.items():
        if progress is not None and post_task is not None:
            progress.update(post_task, description="[red] Creating VRT...")
        vrt_path = os.path.join(base_dir, f"{band_name}.vrt")

        # get maximum bounds from all files
        bounds, xRes, yRes = get_bounds_and_resolution(files)

        nodata_value = 0
        if asset_nodata is not None and asset_nodata[band_name]:
            nodata_value = asset_nodata[band_name]

        # create VRT with highest resolution
        gdal.BuildVRT(vrt_path, files,
                      outputBounds=bounds,
                      xRes=xRes,
                      yRes=yRes,
                      resampleAlg="nearest",
                      outputSRS=target_srs,
                      srcNodata=nodata_value,
                      VRTNodata=nodata_value,
                      addAlpha=False)

        if progress is not None and post_task is not None:
            progress.update(post_task, description="[red] Cropping VRT...")

        output_files.append(vrt_path)

        if progress is not None and post_task is not None:
            progress.update(post_task, description=f"[red] Processed {band_name}", advance=1)

    if progress and post_task:
        progress.remove_task(post_task)

    return output_files


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
    output_dir: str,
    target_year: int,
    target_month:int,
    target_assets: dict[str, str],
    target_srs,
    progress: Progress = None,
    max_workers: int = 5,
):
    """
    download STAC data from Earth Search to create tiff file for each asset (eg, B02, B03) required

    :param stac_url: STAC root URL
    :param collection_id: collection id
    :param geopackage_file_path: path to geopackage file
    :param polygons_layer_name: name of layer polygon layer in geopackage to mask
    :param output_dir: output directory
    :param target_year: target year
    :param target_assets: target assets.
    :param target_srs: target projection CRS
    :param progress: rich progress object
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
        progress.update(stac_task, description=f"[cyan]Preparing to download {len(sentinel_items)} assets", total=len(sentinel_items))

    t2 = time.time()
    logger.debug(f"STAC Item search: {t2 - t1} seconds")

    semaphore = asyncio.Semaphore(max_workers)

    async def download_item(item: SentinelItem, semaphore):
        os.makedirs(output_dir, exist_ok=True)

        async with semaphore:
            # retry max 5 times if any connection error occurs
            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    item.download_assets(download_dir=output_dir, progress=progress)
                    if progress and stac_task:
                        progress.update(stac_task, description=f"[green]Downloaded {item.id}", advance=1)
                    break
                except asyncio.CancelledError:
                    logger.error(f"Download cancelled for {item.id}")
                    raise
                except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    logger.warning(f"Timeout on attempt {attempt} for {item.id}: {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"Failed after {max_retries} attempts: {item.id}")
                except Exception as e:
                    logger.error(f"Download failed for {item.id}: {e}")
                    break

    tasks = [
        asyncio.create_task(download_item(sentinel_item, semaphore), name=f"{sentinel_item.id}")
        for sentinel_item in sentinel_items
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logging.warning("Download interrupted by user. Cancelling tasks...")

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    if os.path.exists(tmp_cutline_path):
        os.remove(tmp_cutline_path)

    t3 = time.time()
    logger.debug(f"STAC Item download: {t3 - t2} seconds")

    if progress and stac_task:
        progress.update(stac_task, description="[magenta]Cropping downloaded assets...")

    output_files = crop_asset_files(
        base_dir=output_dir,
        target_srs=target_srs,
        asset_nodata=sentinel_items[0].asset_nodata,
        progress=progress,
    )

    if progress and stac_task:
        progress.update(stac_task, description="[bold green]Download and crop complete!")
        progress.remove_task(stac_task)

    t4 = time.time()
    logger.debug(f"Download completed: {t4 - t1} seconds")

    return output_files

