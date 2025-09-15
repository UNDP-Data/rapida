import concurrent
import datetime
import os
from concurrent.futures import ProcessPoolExecutor
import threading
import time
from abc import abstractmethod
import psutil
import pystac
import rasterio
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.enums import Resampling
from rasterio.io import DatasetWriter
from rasterio.features import geometry_mask
import numpy as np
from typing import List, Tuple, Optional
import geopandas as gpd
from shapely.geometry import box
from rich.progress import Progress
from rapida.util.setup_logger import setup_logger


logger = setup_logger('rapida')


class PredictionBase(object):

    @property
    def target_bands(self)->List[str]:
        """
        The list of target band names required running prediction model.
        """
        return self._target_bands

    @property
    def component_name(self)->str:
        """
        The name of the prediction component.
        """
        return self._component_name

    @property
    def output_nodata_value(self)->float:
        """
        No data value for the output file of generating
        """
        return self._output_nodata_value

    @property
    def input_nodata_value(self)->float:
        return self._input_nodata_value

    @property
    def cutoff_date_for_harmonize(self)->datetime.datetime:
        """
        Baseline changed date of Sentinel 2. Sentinel 2 had a breaking change since 25 January 2022.

        https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change

        :return datetime.datetime instance
        """
        cutoff = datetime.datetime(2022, 1, 25, tzinfo=datetime.timezone.utc)
        return cutoff

    @property
    def item(self) -> pystac.Item:
        """
        Get STAC Item instance
        :return: pystac.Item instance
        """
        return self._item

    @property
    def process_tile_size(self):
        """
        tile size to process a model iteratively. Default is 512 px.
        """
        return self._process_tile_size
    @process_tile_size.setter
    def process_tile_size(self, value):
        self._process_tile_size = value

    @property
    def process_tile_buffer(self):
        """
        Tile buffer size to process a model. Default is 64 px
        """
        return self._process_tile_buffer
    @process_tile_buffer.setter
    def process_tile_buffer(self, value):
        self._process_tile_buffer = value

    def __init__(self,
                 item: pystac.Item,
                 component_name: str,
                 bands: List[str],
                 output_nodata_value: float,
                 input_nodata_value: float,
                 tile_size: int = 512,
                 tile_buffer: int = 64):
        """
        Constructor

        :param item: pystac.Item instance
        :param component_name: name of the prediction component
        :param bands: list of band names
        :param output_nodata_value: nodata value for outputs
        :param input_nodata_value: nodata value for inputs
        """
        self._item = item
        self._component_name = component_name
        self._target_bands = bands
        self._output_nodata_value = output_nodata_value
        self._input_nodata_value = input_nodata_value
        self.process_tile_size = tile_size
        self.process_tile_buffer = tile_buffer


    def preinference_check(self, img_paths: List) -> Tuple[int, int, str]:
        """
        Perform pre-inference validation and extract image metadata.

        This function checks:
        - That the number of input image paths matches the expected number of target bands.
        - That each provided image path exists and contains only a single band.
        It then identifies the image with the highest resolution (smallest pixel size) to use as the reference for further processing.

        Raises:
            AssertionError: If the number of provided images does not match the expected number of bands.
            FileNotFoundError: If any of the provided file paths does not exist.
            ValueError: If any image contains more than one band.

        :param img_paths: A list of file paths to the input band images.
        :return A tuple containing:
            - column size (int): The width (in pixels) of the reference image.
            - row size (int): The height (in pixels) of the reference image.
            - min_resolution_path (str): The file path of the image with the highest resolution.
        """

        number_bands = len(self.target_bands)

        assert len(img_paths) > 0 and len(img_paths) == number_bands, f"{number_bands} bands are required for {self.component_name}"

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


    def harmonize_to_old(self, data, nodata_value=0) -> np.ndarray:
        """
        Harmonize new Sentinel-2 data to the old baseline by subtracting a fixed offset.
        described at https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change

        :param data: Sentinel-2 data to harmonize
        :param nodata_value: nodata value to exclude from harmonization
        :return: harmonized data. The input data with an offset of 1000 subtracted.
        """
        offset = 1000
        if self.item.datetime >= self.cutoff_date_for_harmonize:
            harmonized = np.where(data != nodata_value, data - offset, data)
            return harmonized
        else:
            return data

    def process_tile(self, row, col,
                     img_paths,
                     min_resolution_path,
                     mask_union=None):
        logger.debug(f"Processing window at row {row}, col {col}")

        with rasterio.open(min_resolution_path) as ref_src:
            ref_col_off = max(col - self.process_tile_buffer, 0)
            ref_row_off = max(row - self.process_tile_buffer, 0)
            ref_col_end = min(col + self.process_tile_size + self.process_tile_buffer, ref_src.width)
            ref_row_end = min(row + self.process_tile_size + self.process_tile_buffer, ref_src.height)

            ref_window = Window(ref_col_off, ref_row_off,
                                ref_col_end - ref_col_off, ref_row_end - ref_row_off)
            ref_height = int(ref_window.height)
            ref_width = int(ref_window.width)
            ref_bounds = rasterio.windows.bounds(ref_window, ref_src.transform)
            ref_transform = rasterio.windows.transform(ref_window, ref_src.transform)

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
                harmonized_data = self.harmonize_to_old(band_data)
                raw_data[:, :, i] = harmonized_data

                # create mask for nodata pixels
                if nodata_mask is None:
                    nodata_value = src.nodata if src.nodata is not None else self.input_nodata_value
                    nodata_mask = (band_data == nodata_value)

        mask_array = None
        if mask_union:
            mask_array = geometry_mask(
                [mask_union],
                out_shape=(ref_height, ref_width),
                transform=ref_transform,
                invert=True
            )
            raw_data[~mask_array] = self.input_nodata_value

        predicted_data = self.run_model(raw_data)

        if mask_array is not None:
            predicted_data[~mask_array] = self.output_nodata_value

        # crop original tile by removing buffer
        start_row = self.process_tile_buffer if row >= self.process_tile_buffer else 0
        start_col = self.process_tile_buffer if col >= self.process_tile_buffer else 0
        end_row = start_row + min(self.process_tile_size, ref_height - start_row)
        end_col = start_col + min(self.process_tile_size, ref_width - start_col)

        original_tile = predicted_data[start_row:end_row, start_col:end_col]

        # The model predicts no data pixel where value is 1,
        # so remove predicted values for nodata mask after predicting
        if nodata_mask is not None:
            mask = nodata_mask[start_row:end_row, start_col:end_col]
            original_tile[mask] = self.output_nodata_value

        return (row, col, original_tile)

    @abstractmethod
    def run_model(self, data: np.ndarray) -> np.ndarray:
        """
        Abstract method for running the prediction model on input data.

        This method must be implemented by subclasses to perform the actual
        prediction or inference logic, such as applying a machine learning model
        or cloud detection algorithm. It takes the prepared and harmonized
        input data and returns the predicted output array.

        :param data: Input data array, typically preprocessed and harmonized, with shape [height, width, bands].
        :return: Output prediction array, typically a 2D array [height, width] or a probability map, depending on the model's purpose.
        """
        raise NotImplementedError

    def write_metadata(self, dst: DatasetWriter) -> DatasetWriter:
        """
        Write additional metadata to the output raster file.

        This method can be overridden by subclasses to add specific metadata,
        such as colormaps, tags, or statistics, to the output raster file.
        By default, it performs no additional action and simply returns the original
        dataset writer object.

        :param dst: An open rasterio dataset writer object, representing the output file to which metadata can be added.
        :return: The same dataset writer object, optionally updated with additional metadata.
        """
        return dst

    def predict(self,
                img_paths: List[str],
                output_file_path: str,
                mask_file: Optional[str] = None,
                mask_layer: Optional[str] = None,
                num_workers: Optional[int] = None,
                progress: Optional[Progress] = None) -> str:
        """
        Perform prediction across image tiles and write the results to an output GeoTIFF file.

        This function divides the input image area into tiles, optionally filters them using a mask,
        runs parallel predictions (using process_tile) across all tiles, and writes the results into
        a GeoTIFF output. It supports multithreading and progress tracking.

        :param img_paths: List of file paths to input band images.
        :param output_file_path: Path to the output GeoTIFF file.
        :param mask_file: Path to a mask vector file (e.g., GeoPackage) to limit the prediction area.
        :param mask_layer: Layer name within the mask file.
        :param num_workers: Number of parallel worker processes. Defaults to physical core count.
        :param progress: rich.progress instance.
        :return: The output GeoTIFF file.
        """
        t1 = time.time()
        predict_task = None
        if progress:
            predict_task = progress.add_task(f"[cyan]Starting {self.component_name} prediction")

        col_size, row_size, min_resolution_path = self.preinference_check(img_paths)

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
            # add 100m buffer for mask polygon
            mask_union = mask_gdf.buffer(100).union_all()

        if predict_task is not None:
            progress.update(predict_task, description=f"[cyan]Creating output file: {output_file_path}")

        # Collect masked tiles only for input files
        tile_jobs = []
        all_cols = []
        all_rows = []

        for row in range(0, row_size, self.process_tile_size):
            for col in range(0, col_size, self.process_tile_size):
                window = Window(col, row, min(self.process_tile_size, col_size - col), min(self.process_tile_size, row_size - row))
                bounds = rasterio.windows.bounds(window, dst_transform)
                tile_geom = box(bounds[0], bounds[1], bounds[2], bounds[3])

                if mask_union.intersects(tile_geom):
                    tile_jobs.append((row, col))
                    all_cols.append(col)
                    all_rows.append(row)

        # Determine bounding box
        min_col = min(all_cols)
        max_col = max(all_cols) + self.process_tile_size
        min_row = min(all_rows)
        max_row = max(all_rows) + self.process_tile_size

        out_width = max_col - min_col
        out_height = max_row - min_row

        # Update transform for cropped area
        cropped_transform = window_transform(Window(min_col, min_row, out_width, out_height), dst_transform)

        with rasterio.open(output_file_path,
                           mode='w',
                           driver="GTiff",
                           dtype='uint8',
                           width=out_width,
                           height=out_height,
                           crs=crs,
                           transform=cropped_transform,
                           nodata=self.output_nodata_value,
                           count=1,
                           compress='ZSTD',
                           tiled=True) as dst:

            self.write_metadata(dst)

            if predict_task:
                progress.update(predict_task, description=f"[cyan]Running prediction for {len(tile_jobs)} tiles",
                                total=len(tile_jobs))

            write_lock = threading.Lock()

            if num_workers is None:
                num_workers = psutil.cpu_count(logical=False)

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                job_iter = iter(tile_jobs)
                running_futures = {}

                # add first batch to process
                for _ in range(num_workers):
                    try:
                        row, col = next(job_iter)
                    except StopIteration:
                        break
                    task_id = progress.add_task(f"[green]Processing tile ({row}, {col})",
                                                total=None) if progress else None
                    fut = executor.submit(self.process_tile, row, col, img_paths, min_resolution_path, mask_union)
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
                            task_id = progress.add_task(f"[green]Processing tile ({row}, {col})",
                                                        total=None) if progress else None
                            fut = executor.submit(self.process_tile, row, col, img_paths, min_resolution_path, mask_union)
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