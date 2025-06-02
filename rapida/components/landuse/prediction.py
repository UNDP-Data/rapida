import concurrent
import datetime
import os
import json
import time

import psutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from concurrent.futures import ProcessPoolExecutor
import threading
from typing import List
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import pystac
import rasterio
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.enums import Resampling
from rapida.components.landuse.constants import DYNAMIC_WORLD_COLORMAP
from rapida.util.setup_logger import setup_logger

logger = setup_logger('rapida')

try:
    import tensorflow as tf
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception as ie:
    tf = None


def normalize(image):
    """
    :param image:
    :return:
    """
    norm_percentiles = np.array([
        [1.7417268007636313, 2.023298706048351],
        [1.7261204997060209, 2.038905204308012],
        [1.6798346251414997, 2.179592821212937],
        [1.7734969472909623, 2.2890068333026603],
        [2.289154079164943, 2.6171674549378166],
        [2.382939712192371, 2.773418590375327],
        [2.3828939530384052, 2.7578332604178284],
        [2.1952484264967844, 2.789092484314204],
        [1.554812948247501, 2.4140534947492487]
    ])

    image = np.log(image * 0.005 + 1)
    image = (image - norm_percentiles[:, 0]) / norm_percentiles[:, 1]
    image = np.exp(image * 5 - 1)
    image = image / (image + 1)
    return image





def load_model():
    """
    Load the model from the saved path
    This is the model from Dynamic World
    """
    folder = os.path.dirname(__file__)

    model_path = os.path.join(folder, 'model', 'forward')

    # model_path = "./model/forward"
    model = tf.saved_model.load(model_path)
    return model

def create_multi_band_image(input_files, output_file):
    """
    Create a multi-band image from the input files
    :param input_files:
    :param output_file:
    :return:
    """
    raw_data = np.empty(shape=(len(input_files),), dtype="u2")
    with rasterio.open(input_files[0]) as src:
        meta = src.meta.copy()
        meta.update(count=len(input_files))

    with rasterio.open(output_file, "w", **meta) as dst:
        for i, file in enumerate(input_files):
            with rasterio.open(file) as src:
                raw_data[:, :, i] = src.read(1)
    logger.debug(f"Multi-band image saved as {output_file}")
    return output_file

def preinference_check(img_paths: List):
    assert len(img_paths) > 0 and len(img_paths) == 9, "9 bands are required for prediction with this model"

    min_pixel_size = float('inf')
    min_resolution_path = None
    col_size = None
    row_size = None

    for img_path in img_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File {img_path} does not exist")
        with rasterio.open(img_path) as src:
            if src.count != 1:
                raise ValueError("Each of the image must have one band")
            transform = src.transform
            pixel_size_x = transform.a
            if pixel_size_x < min_pixel_size:
                min_pixel_size = pixel_size_x
                min_resolution_path = img_path
                col_size = src.width
                row_size = src.height

    return col_size, row_size, min_resolution_path


def harmonize_to_old(data, target_datetime, nodata_value=0):
    """
    Harmonize new Sentinel-2 data to the old baseline by subtracting a fixed offset.
    described at https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change

    :param data: Sentinel-2 data to harmonize
    :param target_datetime: Target datetime of the item
    :param nodata_value: nodata value to exclude from harmonization
    :return: harmonized data. The input data with an offset of 1000 subtracted.
    """
    cutoff = datetime.datetime(2022, 1, 25, tzinfo=datetime.timezone.utc)
    offset = 1000
    if target_datetime >= cutoff:
        harmonized = np.where(data != nodata_value, data - offset, data)
        return harmonized
    else:
        return data

def process_tile(row, col,
                 img_paths,
                 min_resolution_path,
                 target_datetime,
                 buffer,
                 tile_size=256,
                 landuse_nodata=255):
    logger.debug(f"Processing window at row {row}, col {col}")

    with rasterio.open(min_resolution_path) as ref_src:
        ref_col_off = max(col - buffer, 0)
        ref_row_off = max(row - buffer, 0)
        ref_col_end = min(col + tile_size + buffer, ref_src.width)
        ref_row_end = min(row + tile_size + buffer, ref_src.height)

        ref_window = Window(ref_col_off, ref_row_off,
                            ref_col_end - ref_col_off, ref_row_end - ref_row_off)
        ref_height = int(ref_window.height)
        ref_width = int(ref_window.width)
        ref_bounds = rasterio.windows.bounds(ref_window, ref_src.transform)

    raw_data = np.empty((ref_height, ref_width, len(img_paths)), dtype="u2")

    nodata_mask = None

    for i, file_path in enumerate(img_paths):
        with rasterio.open(file_path) as src:
            left, bottom, right, top = ref_bounds
            src_window = from_bounds(
                left, bottom, right, top,
                transform=src.transform
            ).round_offsets().round_lengths()

            band_data = src.read(
                1,
                window=src_window,
                out_shape=(ref_height, ref_width),
                resampling=Resampling.bilinear
            )
            harmonized_data = harmonize_to_old(band_data, target_datetime)
            raw_data[:, :, i] = harmonized_data

            # create mask for nodata pixels
            if nodata_mask is None and src.nodata is not None:
                nodata_mask = (band_data == src.nodata)

    normalized_data = normalize(raw_data)
    nhwc_image = tf.expand_dims(tf.cast(normalized_data, dtype=tf.float32), axis=0)
    logger.debug("Running model inference")
    forward_model = load_model()
    lulc_logits = forward_model(nhwc_image)
    lulc_prob = tf.nn.softmax(lulc_logits)
    lulc_prob = np.array(lulc_prob[0])
    lulc_prediction = np.argmax(lulc_prob, axis=-1)

    # crop original tile by removing buffer
    start_row = buffer if row >= buffer else 0
    start_col = buffer if col >= buffer else 0
    end_row = start_row + min(tile_size, ref_height - start_row)
    end_col = start_col + min(tile_size, ref_width - start_col)

    original_tile = lulc_prediction[start_row:end_row, start_col:end_col]

    # The model predicts no data pixel where value is zero as 1 (trees),
    # so remove predicted values for nodata mask after predicting
    if nodata_mask is not None:
        mask = nodata_mask[start_row:end_row, start_col:end_col]
        original_tile[mask] = landuse_nodata

    return (row, col, original_tile)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def predict(img_paths: List[str],
            output_file_path: str,
            item: pystac.Item,
            mask_file=None,
            mask_layer=None,
            num_workers=None,
            progress = None):
    t1 = time.time()
    predict_task = None
    if progress:
        predict_task = progress.add_task(f"[cyan]Starting land use prediction")

    col_size, row_size, min_resolution_path = preinference_check(img_paths)
    tile_size=1024
    buffer = 64
    landuse_nodata = 255

    if predict_task is not None:
        progress.update(predict_task, description="[cyan]Opening first image to get metadata")

    with rasterio.open(min_resolution_path) as src:
        crs = src.crs
        dst_transform = src.transform

    # Load mask if provided
    mask_union = None
    if mask_file and mask_layer:
        gdf = gpd.read_file(mask_file, layer=mask_layer)
        mask_gdf = gdf.to_crs(crs)
        mask_union = mask_gdf.union_all()

    if predict_task is not None:
        progress.update(predict_task, description=f"[cyan]Creating output file: {output_file_path}")

    # Collect masked tiles only
    tile_jobs = []
    all_cols = []
    all_rows = []

    for row in range(0, row_size, tile_size):
        for col in range(0, col_size, tile_size):
            window = Window(col, row, min(tile_size, col_size - col), min(tile_size, row_size - row))
            bounds = rasterio.windows.bounds(window, dst_transform)
            tile_geom = box(bounds[0], bounds[1], bounds[2], bounds[3])

            if mask_union.intersects(tile_geom):
                tile_jobs.append((row, col))
                all_cols.append(col)
                all_rows.append(row)

    # Determine bounding box
    min_col = min(all_cols)
    max_col = max(all_cols) + tile_size
    min_row = min(all_rows)
    max_row = max(all_rows) + tile_size

    out_width = max_col - min_col
    out_height = max_row - min_row

    # Update transform for cropped area
    cropped_transform = window_transform(Window(min_col, min_row, out_width, out_height), dst_transform)

    landuse_colormap = {k: hex_to_rgb(v['color']) for k, v in DYNAMIC_WORLD_COLORMAP.items()}
    statistics_unique_values = {str(k): v['label'] for k, v in DYNAMIC_WORLD_COLORMAP.items()}
    unique_values_json = json.dumps(statistics_unique_values, ensure_ascii=False)

    with rasterio.open(output_file_path,
                       mode='w',
                       driver="GTiff",
                       dtype='uint8',
                       width=col_size,
                       height=row_size,
                       crs=crs,
                       transform=cropped_transform,
                       nodata=landuse_nodata,
                       count=1,
                       compress='ZSTD',
                       tiled=True,
                       photometric='palette') as dst:
        # write colormap
        dst.write_colormap(1, landuse_colormap)
        # write STATISTICS_UNIQUE_VALUES for GeoHub
        dst.update_tags(1, STATISTICS_UNIQUE_VALUES=unique_values_json)

        statistics_unique_values = {
            str(k): v['label'] for k, v in DYNAMIC_WORLD_COLORMAP.items()
        }
        unique_values_json = json.dumps(statistics_unique_values, ensure_ascii=False)
        dst.update_tags(1, STATISTICS_UNIQUE_VALUES=unique_values_json)

        if predict_task:
            progress.update(predict_task, description=f"[cyan]Running prediction for {len(tile_jobs)} tiles", total=len(tile_jobs))

        write_lock = threading.Lock()

        if num_workers is None:
            num_workers = psutil.cpu_count(logical=False)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            job_iter = iter(tile_jobs)
            running_futures = {}

            # add first batch to process
            for _ in range(num_workers):
                row, col = next(job_iter)
                task_id = progress.add_task(f"[green]Processing tile ({row}, {col})", total=None) if progress else None
                fut = executor.submit(process_tile, row, col, img_paths, min_resolution_path, item.datetime,
                                      buffer, tile_size, landuse_nodata)
                running_futures[fut] = (row, col, task_id)

            while running_futures:
                # wait for any future to complete
                done, _ = concurrent.futures.wait(running_futures, return_when=concurrent.futures.FIRST_COMPLETED)

                for fut in done:
                    row, col, task_id = running_futures.pop(fut)
                    row, col, tile_data = fut.result()

                    tile_height = tile_data.shape[0]
                    tile_width = tile_data.shape[1]

                    write_window = Window(col - min_col, row - min_row, tile_width, tile_height)

                    with write_lock:
                        dst.write(tile_data.astype(np.float32), 1, window=write_window)

                    if progress:
                        if predict_task:
                            progress.update(predict_task, advance=1)
                        if task_id:
                            progress.remove_task(task_id)

                    # add next job
                    try:
                        row, col = next(job_iter)
                        task_id = progress.add_task(f"[green]Processing tile ({row}, {col})", total=None) if progress else None
                        fut = executor.submit(process_tile, row, col, img_paths, min_resolution_path, item.datetime,
                                              buffer, tile_size, landuse_nodata)
                        running_futures[fut] = (row, col, task_id)
                    except KeyboardInterrupt:
                        logger.info("Prediction interrupted by user. Cancelling tasks..")
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise
                    except StopIteration:
                        continue

    t2 = time.time()
    logger.debug(f"total time of prediction: {t2 - t1}")

    if predict_task is not None:
        progress.update(predict_task, description=f"[cyan]Prediction process completed successfully")
        progress.remove_task(predict_task)
    return output_file_path



if __name__ == "__main__":
    pass