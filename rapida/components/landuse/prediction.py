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
import pystac
import rasterio
from rasterio.windows import Window
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
    col_sizes = []
    row_sizes = []
    for img_path in img_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File {img_path} does not exist")
        with rasterio.open(img_path) as src:
            if src.count != 1:
                raise ValueError("Each of the image must have one band")
            col_sizes.append(src.width)
            row_sizes.append(src.height)
    if len(set(col_sizes)) != 1:
        raise ValueError("All images must have the same dimensions")
    if len(set(row_sizes)) != 1:
        raise ValueError("All images must have the same dimensions")
    return col_sizes[0], row_sizes[0]


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
                 target_datetime,
                 buffer,
                 row_size,
                 col_size,
                 tile_size=256,
                 landuse_nodata=255):
    logger.debug(f"Processing window at row {row}, col {col}")

    # create buffer (64px x 64px) for original tile size
    row_off = max(row - buffer, 0)
    col_off = max(col - buffer, 0)

    window_height = min(row + tile_size + buffer, row_size) - row_off
    window_width = min(col + tile_size + buffer, col_size) - col_off

    window = Window(col_off, row_off, window_width, window_height)
    raw_data = np.empty((window_height, window_width, 9), dtype="u2")

    nodata_mask = None

    for i, file_path in enumerate(img_paths):
        with rasterio.open(file_path) as src:
            band_data = src.read(1, window=window)
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
    start_row = row - row_off
    start_col = col - col_off
    end_row = start_row + min(tile_size, row_size - row)
    end_col = start_col + min(tile_size, col_size - col)

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
            num_workers=None,
            progress = None):
    t1 = time.time()
    predict_task = None
    if progress:
        predict_task = progress.add_task(f"[cyan]Starting land use prediction")

    col_size, row_size = preinference_check(img_paths)
    tile_size=1024
    buffer = 64
    landuse_nodata = 255

    if predict_task is not None:
        progress.update(predict_task, description="[cyan]Opening first image to get metadata")

    with rasterio.open(img_paths[0]) as src:
        crs = src.crs
        dst_transform = src.transform

    if predict_task is not None:
        progress.update(predict_task, description=f"[cyan]Creating output file: {output_file_path}")

    with rasterio.open(output_file_path,
                       mode='w',
                       driver="GTiff",
                       dtype='uint8',
                       width=col_size,
                       height=row_size,
                       crs=crs,
                       transform=dst_transform,
                       nodata=landuse_nodata,
                       count=1,
                       compress='ZSTD',
                       tiled=True,
                       photometric='palette') as dst:
        # write colormap
        landuse_colormap = {k: hex_to_rgb(v['color']) for k, v in DYNAMIC_WORLD_COLORMAP.items()}
        dst.write_colormap(1, landuse_colormap)

        # write STATISTICS_UNIQUE_VALUES for GeoHub
        statistics_unique_values = {
            str(k): v['label'] for k, v in DYNAMIC_WORLD_COLORMAP.items()
        }
        unique_values_json = json.dumps(statistics_unique_values, ensure_ascii=False)
        dst.update_tags(1, STATISTICS_UNIQUE_VALUES=unique_values_json)

        tile_jobs = [
            (row, col)
            for row in range(0, row_size, tile_size)
            for col in range(0, col_size, tile_size)
        ]

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
                fut = executor.submit(process_tile, row, col, img_paths, item.datetime,
                                      buffer, row_size, col_size,
                                      tile_size, landuse_nodata)
                running_futures[fut] = (row, col, task_id)

            while running_futures:
                # wait for any future to complete
                done, _ = concurrent.futures.wait(running_futures, return_when=concurrent.futures.FIRST_COMPLETED)

                for fut in done:
                    row, col, task_id = running_futures.pop(fut)
                    row, col, original_tile = fut.result()

                    with write_lock:
                        dst.write(original_tile.astype(np.uint8), 1,
                                  window=Window(col, row, min(tile_size, col_size - col), min(tile_size, row_size - row)))

                    if progress:
                        if predict_task:
                            progress.update(predict_task, advance=1)
                        if task_id:
                            progress.remove_task(task_id)

                    # add next job
                    try:
                        row, col = next(job_iter)
                        task_id = progress.add_task(f"[green]Processing tile ({row}, {col})", total=None) if progress else None
                        fut = executor.submit(process_tile, row, col, img_paths, item.datetime,
                                              buffer, row_size, col_size,
                                              tile_size, landuse_nodata)
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