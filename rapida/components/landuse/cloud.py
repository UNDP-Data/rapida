import concurrent
import datetime
import os
import time
import psutil
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
from s2cloudless import S2PixelCloudDetector, download_bands_and_valid_data_mask
from rich.progress import Progress
from rapida.util.setup_logger import setup_logger


logger = setup_logger('rapida')


def preinference_check(img_paths: List):
    assert len(img_paths) > 0 and len(img_paths) == 10, "10 bands are required for cloud detection"

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


def harmonize_to_old(data, target_datetime, reflectance_conversion_factor, nodata_value=0):
    """
    Harmonize new Sentinel-2 data to the old baseline.

    ESA updated the Sentinel-2 processing baseline to version 04.00 in January, 2022, which introduced breaking changes to the interpretation of digital numbers (DN).

    It is described at:

    - https://github.com/sentinel-hub/sentinel2-cloud-detector?tab=readme-ov-file#input-sentinel-2-scenes
    - https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/#harmonize-values

    :param data: Sentinel-2 data to harmonize
    :param target_datetime: Target datetime of the item
    :param nodata_value: nodata value to exclude from harmonization
    :return: harmonized data. The input data with an offset of 1000 subtracted.
    """
    cutoff = datetime.datetime(2022, 1, 25, tzinfo=datetime.timezone.utc)
    reflectance_factor = 10000
    offset = 1000
    if target_datetime >= cutoff:
        harmonized = np.where(data != nodata_value, (data - offset).astype(np.float32) * reflectance_conversion_factor/ reflectance_factor, data)
        harmonized = np.clip(harmonized, 0, None)
        return harmonized
    else:
        harmonized = np.where(data != nodata_value, data.astype(np.float32) * reflectance_conversion_factor / reflectance_factor, data)
        harmonized = np.clip(harmonized, 0, None)
        return harmonized


def process_tile(row, col,
                 img_paths,
                 min_resolution_path,
                 item: pystac.Item,
                 buffer,
                 tile_size=256):
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

            raw_data[:, :, i] = band_data

    reflectance_conversion_factor = item.properties.get("s2:reflectance_conversion_factor")
    raw_data = harmonize_to_old(raw_data, item.datetime, reflectance_conversion_factor)

    cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=8, dilation_size=2, all_bands=False)
    cloud_prob = cloud_detector.get_cloud_probability_maps(raw_data[np.newaxis, ...])
    logger.debug(f"Cloud detection probability: {cloud_prob}")
    cloud_mask = cloud_detector.get_mask_from_prob(cloud_prob)

    # crop original tile by removing buffer
    start_row = buffer if row >= buffer else 0
    start_col = buffer if col >= buffer else 0
    end_row = start_row + min(tile_size, ref_height - start_row)
    end_col = start_col + min(tile_size, ref_width - start_col)

    original_tile = cloud_mask[0][start_row:end_row, start_col:end_col]

    return (row, col, original_tile)


def cloud_detect(img_paths: List[str],
                 output_file_path: str,
                 item: pystac.Item,
                 mask_file=None,
                 mask_layer=None,
                 num_workers=None,
                 progress = None):
    t1 = time.time()
    predict_task = None
    if progress:
        predict_task = progress.add_task(f"[cyan]Starting cloud detection")

    col_size, row_size, min_resolution_path = preinference_check(img_paths)
    tile_size=1024
    buffer = 64
    no_data_value = 9999

    logger.debug(f"min_resolution_path: {min_resolution_path}")

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

    with rasterio.open(output_file_path,
                       mode='w',
                       driver="GTiff",
                       dtype='float32',
                       width=col_size,
                       height=row_size,
                       crs=crs,
                       transform=cropped_transform,
                       nodata=no_data_value,
                       count=1,
                       compress='ZSTD',
                       tiled=True) as dst:

        if predict_task:
            progress.update(predict_task, description=f"[cyan]Running cloud detection for {len(tile_jobs)} tiles", total=len(tile_jobs))

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
                fut = executor.submit(process_tile, row, col, img_paths, min_resolution_path, item,
                                      buffer, tile_size)
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
                        dst.update_stats()

                    if progress:
                        if predict_task:
                            progress.update(predict_task, advance=1)
                        if task_id:
                            progress.remove_task(task_id)

                    # add next job
                    try:
                        row, col = next(job_iter)
                        task_id = progress.add_task(f"[green]Processing tile ({row}, {col})", total=None) if progress else None
                        fut = executor.submit(process_tile, row, col, img_paths, min_resolution_path, item,
                                              buffer, tile_size)
                        running_futures[fut] = (row, col, task_id)
                    except KeyboardInterrupt:
                        logger.info("Cloud detection interrupted by user. Cancelling tasks..")
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise
                    except StopIteration:
                        continue

    t2 = time.time()
    logger.debug(f"total time of cloud detection: {t2 - t1}")

    if predict_task is not None:
        progress.update(predict_task, description=f"[cyan]Cloud detection process completed successfully")
        progress.remove_task(predict_task)
    return output_file_path



if __name__ == "__main__":
    img_paths = ['/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B01.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B02.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B04.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B05.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B08.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B8A.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B09.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B10.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B11.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B12.jp2']
    output_file_path = "/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/cloud.tif"
    item_url = '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/item.json'
    mask_file = "/data/kigali_small/data/kigali_small.gpkg"
    mask_layer = "polygons"

    item = pystac.Item.from_file(item_url)

    with Progress() as progress:
        t1 = time.time()
        cloud_detect(img_paths=img_paths, output_file_path=output_file_path, item=item, progress=progress, mask_file=mask_file, mask_layer=mask_layer,)
        t2 = time.time()

        logger.info(f"prediction time: {t2 - t1}")


