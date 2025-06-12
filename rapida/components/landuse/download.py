import logging
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime
from queue import Queue
from osgeo import gdal
from rich.progress import Progress
import geopandas as gpd
import time

from rapida.components.landuse.sentinel_item import SentinelItem
from rapida.components.landuse.stac_collection import StacCollection

logger = logging.getLogger('rapida')


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
    target_srs,
    datetime_range:str,
    cloud_cover: int = 5,
    progress: Progress = None,
    max_workers: int = 1,
):
    """
    download STAC data from Earth Search to create tiff file for each asset (eg, B02, B03) required

    :param stac_url: STAC root URL
    :param collection_id: collection id
    :param geopackage_file_path: path to geopackage file
    :param polygons_layer_name: name of layer polygon layer in geopackage to mask
    :param output_file: output file path
    :param target_srs: target projection CRS
    :param datetime_range: datetime range for searching. Format is yyyy-mm-dd/yyyy-mm-dd. Default is 12 months ending today's date
    :param cloud_cover: how much minimum cloud cover rate to search for. Default is 5.
    :param progress: rich progress object
    :param max_workers: maximum number of workers to download JP2 file concurrently
    :return the list of output files
    """
    t1 = time.time()

    stac_task = None
    if progress:
        stac_task = progress.add_task(f"[cyan]Connecting to {stac_url}/{collection_id}", total=None)

    stac_collection = StacCollection(stac_url=stac_url,
                                     mask_file=geopackage_file_path,
                                     mask_layer=polygons_layer_name)

    if progress and stac_task:
        progress.update(stac_task, description="[yellow]Searching for STAC items...")

    latest_per_tile = stac_collection.search_items(
                      collection_id=collection_id,
                      datetime_range=datetime_range,
                      cloud_cover=cloud_cover,
                      progress=progress,)

    if len(latest_per_tile.values()) == 0:
        if progress and stac_task:
            progress.remove_task(stac_task)
        raise RuntimeError(
            f"No items found from Sentinel 2. Try to set wider date range or increase cloud cover rate.")

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

            progress.update(predict_task, description=f"[cyan]Creating cloud mask {item.item.id}")
            item.detect_cloud(progress=progress)

            progress.update(predict_task, description=f"[cyan]Predicting landuse {item.item.id}")
            temp_predict_file = item.predict(progress=progress)

            progress.update(predict_task, description=f"[cyan]Removing cloud from landuse {item.item.id}")
            item.mask_cloud_pixels(temp_predict_file, item.predicted_file, progress=progress)

            if os.path.exists(temp_predict_file):
                os.remove(temp_predict_file)

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

