import concurrent
import os
import time
import psutil
from concurrent.futures import ProcessPoolExecutor
import threading
from typing import List
import numpy as np
import rasterio
from rasterio.windows import Window
from s2cloudless import S2PixelCloudDetector, download_bands_and_valid_data_mask
from rich.progress import Progress
from rapida.util.setup_logger import setup_logger


logger = setup_logger('rapida')


def preinference_check(img_paths: List):
    assert len(img_paths) > 0 and len(img_paths) == 10, "10 bands are required for cloud detection"
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


def process_tile(row, col, img_paths, buffer, row_size, col_size, tile_size=256, scale=0.0001):
    logger.debug(f"Processing window at row {row}, col {col}")

    # create buffer (64px x 64px) for original tile size
    row_off = max(row - buffer, 0)
    col_off = max(col - buffer, 0)

    window_height = min(row + tile_size + buffer, row_size) - row_off
    window_width = min(col + tile_size + buffer, col_size) - col_off

    window = Window(col_off, row_off, window_width, window_height)

    raw_data = np.empty((window_height, window_width, 10), dtype="u2")

    for i, file_path in enumerate(img_paths):
        with rasterio.open(file_path) as src:
            band_data = src.read(1, window=window)
            raw_data[:, :, i] = band_data

    raw_data = raw_data.astype(np.float32) * scale

    cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=False)
    cloud_mask = cloud_detector.get_cloud_masks(raw_data[np.newaxis, ...])

    # crop original tile by removing buffer
    start_row = row - row_off
    start_col = col - col_off
    end_row = start_row + min(tile_size, row_size - row)
    end_col = start_col + min(tile_size, col_size - col)

    original_tile = cloud_mask[0][start_row:end_row, start_col:end_col]

    return (row, col, original_tile)


def cloud_detect(img_paths: List[str],
                 output_file_path: str,
                 num_workers=None,
                 progress = None):
    t1 = time.time()
    predict_task = None
    if progress:
        predict_task = progress.add_task(f"[cyan]Starting cloud detection")

    col_size, row_size = preinference_check(img_paths)
    tile_size=1024
    buffer = 64

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
                       dtype='float32',
                       width=col_size,
                       height=row_size,
                       crs=crs,
                       transform=dst_transform,
                       count=1,
                       compress='ZSTD',
                       tiled=True) as dst:

        tile_jobs = [
            (row, col)
            for row in range(0, row_size, tile_size)
            for col in range(0, col_size, tile_size)
        ]

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
                fut = executor.submit(process_tile, row, col, img_paths,
                                      buffer, row_size, col_size,
                                      tile_size)
                running_futures[fut] = (row, col, task_id)

            while running_futures:
                # wait for any future to complete
                done, _ = concurrent.futures.wait(running_futures, return_when=concurrent.futures.FIRST_COMPLETED)

                for fut in done:
                    row, col, task_id = running_futures.pop(fut)
                    row, col, original_tile = fut.result()

                    with write_lock:
                        dst.write(original_tile.astype(np.float32), 1,
                                  window=Window(col, row, min(tile_size, col_size - col), min(tile_size, row_size - row)))
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
                        fut = executor.submit(process_tile, row, col, img_paths,
                                              buffer, row_size, col_size,
                                              tile_size)
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
    # img_paths = ['/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/B01.tif', '/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/B02.tif',
    #              '/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/B04.tif', '/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/B05.tif',
    #              '/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/B08.tif', '/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/B8A.tif',
    #              '/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/B09.tif', '/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/B10.tif',
    #              '/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/B11.tif', '/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/B12.tif']
    # output_file_path = "/data/sentinel_tests/S2C_35MRT_20250225_0_L1C/cloud.tif"

    img_paths = ['/data/sentinel_tests/MGRS-35MRT/B01.tif', '/data/sentinel_tests/MGRS-35MRT/B02.tif',
                 '/data/sentinel_tests/MGRS-35MRT/B04.tif', '/data/sentinel_tests/MGRS-35MRT/B05.tif',
                 '/data/sentinel_tests/MGRS-35MRT/B08.tif', '/data/sentinel_tests/MGRS-35MRT/B8A.tif',
                 '/data/sentinel_tests/MGRS-35MRT/B09.tif', '/data/sentinel_tests/MGRS-35MRT/B10.tif',
                 '/data/sentinel_tests/MGRS-35MRT/B11.tif', '/data/sentinel_tests/MGRS-35MRT/B12.tif']

    output_file_path="/data/sentinel_tests/MGRS-35MRT/cloud_2.tif"

    with Progress() as progress:
        t1 = time.time()
        cloud_detect(img_paths=img_paths, output_file_path=output_file_path, progress=progress)
        t2 = time.time()

        logger.info(f"prediction time: {t2 - t1}")


