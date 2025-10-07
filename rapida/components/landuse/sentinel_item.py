import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Dict
import shutil
import geopandas as gpd
import numpy as np
import pystac
import rasterio
from osgeo import ogr, osr, gdal
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform
from rich.progress import Progress

from rapida.components.landuse.constants import SENTINEL2_ASSET_MAP
from rapida.components.landuse.prediction.cloud import CloudDetection
from rapida.components.landuse.prediction.landuse import LandusePrediction
from rapida.util.download_remote_file import download_remote_files


logger = logging.getLogger(__name__)


def gdal_callback(complete, message, data):
    if data:
        progressbar, task, timeout_event = data
        if progressbar is not None and task is not None:
            progressbar.update(task, completed=int(complete * 100))
        if timeout_event and timeout_event.is_set():
            logger.info(f'GDAL was signalled to stop...')
            return 0


class SentinelItem(object):
    """
    A class to manage data manipulation for a Sentinel 2 L1C item.
    """

    @property
    def item(self)->pystac.Item:
        """
        Get STAC Item instance
        :return: pystac.Item instance
        """
        return self._item

    @item.setter
    def item(self, item:pystac.Item):
        """
        Set STAC Item instance
        :param item: pystac.Item
        """
        self._item = item

    @property
    def id(self)->str:
        """
        ID for the class instance. It uses item ID of STAC
        """
        return self._item.id

    @property
    def target_asset(self) -> dict[str, str]:
        """
        Dictionary of Earth search asset name and band name
        """
        return self._target_asset

    @property
    def target_srs(self) -> osr.SpatialReference:
        """
        Get target spatial reference used in the mask layer
        """
        if self._target_srs is None:
            with ogr.Open(self.mask_file) as ds:
                if ds is None:
                    raise RuntimeError(f"Could not open GeoPackage: {self.mask_file}")
                layer = ds.GetLayerByName(self.mask_layer)
                if layer is None:
                    raise RuntimeError(f"Layer '{self.mask_layer}' not found in GeoPackage")
                srs = layer.GetSpatialRef()
                if srs is None:
                    raise RuntimeError(f"No CRS found in layer '{self.mask_layer}'")
                self._target_srs = srs
        return self._target_srs

    @property
    def min_resolution(self) -> int:
        """
        Get minimum resolution of assets from item JSON object
        """
        resolutions = []
        for asset_key in self.target_asset:
            asset = self.item.assets[asset_key]
            bands = asset.to_dict().get("raster:bands", [])
            if bands:
                res = bands[0].get("spatial_resolution")
                if res:
                    resolutions.append(res)
        if not resolutions:
            raise ValueError("No spatial resolutions found in item assets.")
        return min(resolutions)

    @property
    def asset_nodata(self) -> dict[str, int]:
        """
        Get no data value for asset from item JSON object
        """
        nodata_dict = {}
        for asset_key, band_name in self.target_asset.items():
            asset = self.item.assets.get(asset_key)
            if not asset:
                nodata_dict[band_name] = 0
                continue

            bands = asset.to_dict().get("raster:bands", [])
            if bands and "nodata" in bands[0]:
                nodata_dict[band_name] = bands[0]["nodata"]
            else:
                nodata_dict[band_name] = 0
        return nodata_dict


    @property
    def asset_files(self) -> dict[str, str]:
        """
        The dictionary of file paths to assets. If download_assets function is not called, it returns empty dict.
        """
        if len(self._asset_files) == 0:
            return {}
        # sorted_dict = dict(sorted(self._asset_files.items(), key=lambda x: int(x[0][1:])))
        return self._asset_files


    @property
    def predicted_file(self)->str:
        """
        Get the file path of the predicted file. If download_assets function is not called, it returns empty string.
        """
        if len(self.asset_files) == 0:
            return ""
        predict_file = os.path.join(os.path.dirname(list(self.asset_files.values())[0]), "landuse_prediction.tif")
        return predict_file

    @property
    def cloud_mask_file(self) -> str:
        """
        Get the file path of the cloud mask file. If download_assets function is not called, it returns empty string.
        """
        if len(self.asset_files) == 0:
            return ""
        cloud_mask_file = os.path.join(os.path.dirname(list(self.asset_files.values())[0]), "cloud_mask.tif")
        return cloud_mask_file


    def __init__(self, item: pystac.Item, mask_file: str, mask_layer: str):
        """
        Constructor

        :param item: pystac.Item instance
        :param mask_file: GPKG path for clipping
        :param mask_layer: layer name to clip by
        """
        self.item = item
        self._target_asset = SENTINEL2_ASSET_MAP
        self._target_srs = None
        self._asset_files = {}
        self.mask_file = mask_file
        self.mask_layer = mask_layer
        if not self.is_valid_item():
            # if item does not have all required assets, raise error.
            raise RuntimeError(f"This STAC item does not contain required assets.")

    def __del__(self):
        self.__target_srs = None

    def is_valid_item(self):
        """
        Validate whether this STAC item contains all required assets
        """
        return all(asset in self.item.assets for asset in self.target_asset)

    def download_assets(self,
                        download_dir: str,
                        progress=None,
                        force=False) -> Dict[str, str]:
        """
        Download all required bands for this STAC item in parallel.

        :param download_dir: directory to store downloaded GeoTIFFs
        :param progress: optional rich progress bar
        :return: Dictionary of band name -> downloaded file path
        """
        self._asset_files = {
            band_name: os.path.join(download_dir, self.id, f"{band_name}.jp2")
            for asset_key, band_name in self.target_asset.items()
        }

        cloud_cover = self.item.properties.get('eo:cloud_cover', None)
        if cloud_cover is not None and cloud_cover < 1.0:
            for band_name in list(self._asset_files.keys()):
                if band_name not in LandusePrediction.required_bands:
                    self._asset_files.pop(band_name, None)

                    # Remove keys in target_asset where value == band_name
                    keys_to_remove = [k for k, v in self.target_asset.items() if v == band_name]
                    for k in keys_to_remove:
                        self.target_asset.pop(k, None)

        item_path = os.path.join(download_dir, self.id, "item.json")
        """
            I really question the lines below. The decision to skip download needs to be made based on the size 
            in bytes of the existing local file and the remote one. The force flag is sued to skip this decision and \
            just overwrite the file. 
        """
        # if not force and os.path.exists(item_path):
        #     with open(item_path, "r", encoding="utf-8") as f:
        #         existing_item = json.load(f)
        #         if existing_item.get("id") == self.item.id:
        #             logger.warning(f"The same item ({self.id}) has already been downloaded. Skipping")
        #             return self.asset_files
        #         else:
        #             # if different item id, delete old prediction file if exists
        #             if os.path.exists(self.predicted_file):
        #                 os.remove(self.predicted_file)
        #             if os.path.exists(self.cloud_mask_file):
        #                 os.remove(self.cloud_mask_file)

        file_urls = []
        url_to_target_mapping = {}
        for asset_key, band_name in self.target_asset.items():
            asset = self.item.assets[asset_key]
            url = self._s3_to_http(asset.href)
            file_urls.append(url)
            band_dir = os.path.join(download_dir, self.id)
            file_extension = os.path.splitext(url)[1]
            target_path = os.path.join(band_dir, f"{band_name}{file_extension}")
            url_to_target_mapping[url] = target_path

        def get_target_path_for_url(url, dst_folder):
            target_path = url_to_target_mapping.get(url)
            return target_path

        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                download_remote_files(
                    file_urls=file_urls,
                    dst_folder=download_dir,
                    target_path_func=get_target_path_for_url,
                    progress=progress, force=force
                )
            )

            # once all downloads are successfully done, write item.json in the folder
            # While item.json not real imagery data it contains some extracted medatada which can be usewful
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(self.item.to_dict(), f, indent=2)

            return self.asset_files
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.error(f'Download was cancelled. Removing {download_dir}')
            if os.path.exists(download_dir) and force:shutil.rmtree(download_dir)
            raise
        except Exception as e:
            logger.error(f"Failed to download assets for item {self.id}: {e}")
            raise


    def predict(self, progress=None, **kwargs)->str:
        """
        Predict land use from downloaded assets

        :return: output file path
        """

        landuse_prediction = LandusePrediction(item=self.item)

        temp_file = f"{self.predicted_file}.tmp"
        try:

            img_paths = [self.asset_files[band] for band in landuse_prediction.target_bands]
            landuse_prediction.predict(img_paths=img_paths,
                                       output_file_path=temp_file,
                                       mask_file=self.mask_file,
                                       mask_layer=self.mask_layer,
                                       progress=progress,**kwargs)

            return temp_file
        except Exception as e:
            # delete incomplete predicted file if error occurs
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(self.predicted_file):
                os.remove(self.predicted_file)
            raise e

    def detect_cloud(self, force=False, progress=None)->str:
        """
        Detect cloud from downloaded assets

        :param force: force to do cloud detection even if cloud cover is less than 1 percent
        :return: output file path
        """
        # Check cloud cover percentage from item properties
        cloud_cover = self.item.properties.get('eo:cloud_cover', None)

        if force == False and cloud_cover is not None and cloud_cover < 1.0:
            logger.warning(f"{self.id}: Cloud cover is {cloud_cover}% (< 1%). Skipping cloud detection.")
            return self.cloud_mask_file

        cloud_detection = CloudDetection(item=self.item)

        try:
            img_paths = [self.asset_files[band] for band in cloud_detection.target_bands]

            cloud_detection.predict(img_paths=img_paths,
                                    output_file_path=self.cloud_mask_file,
                                    mask_file=self.mask_file,
                                    mask_layer=self.mask_layer,
                                    progress=progress,)
            return self.cloud_mask_file
        except Exception as e:
            # delete incomplete predicted file if error occurs
            if os.path.exists(self.cloud_mask_file):
                os.remove(self.cloud_mask_file)
            raise e

    def mask_cloud_pixels(self, input_file: str, output_file: str, progress=None)-> str:
        """
        Mask cloud pixels from land use prediction image

        :param input_file: input file path of land use prediction image
        :param output_file: output file path of mask prediction image
        :return: output file path of mask prediction image
        """
        if not os.path.exists(self.cloud_mask_file):
            return self._reproject_data(input_file, output_file)

        mask_task = None
        if progress:
            mask_task = progress.add_task(f"[cyan]Starting masking cloud from land use prediction image...", total=None)

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            temp_masked_file = tmp.name

        try:
            if mask_task is not None:
                progress.update(mask_task, description="[cyan]Applying cloud mask...")
            self._apply_cloud_mask(input_file, temp_masked_file)
            if mask_task is not None:
                progress.update(mask_task, description="[cyan]Reprojecting land use...")
            self._reproject_data(temp_masked_file, output_file)
        except Exception as e:
            raise e
        finally:
            if os.path.exists(temp_masked_file):
                os.unlink(temp_masked_file)
            if mask_task is not None:
                progress.remove_task(mask_task)
        return output_file

    def _apply_cloud_mask(self, input_file: str, output_file: str) -> None:
        """
        Apply cloud mask to input raster without reprojection

        :param input_file: input file path
        :param output_file: output file path for masked data
        """
        # Get mask geometry for both files
        gdf = gpd.read_file(self.mask_file, layer=self.mask_layer)

        with rasterio.open(input_file) as pred_src:
            # Project gdf to match prediction CRS
            gdf = gdf.to_crs(pred_src.crs)
            nodata_value = pred_src.nodata if pred_src.nodata is not None else 255
            colormap = pred_src.colormap(1)
            band_tags = pred_src.tags(1)

            # Crop prediction data to mask geometry
            pred_crop, pred_transform = mask(pred_src, gdf.geometry, crop=True, filled=True, nodata=nodata_value)

            profile = pred_src.profile.copy()

        with rasterio.open(self.cloud_mask_file) as cloud_src:
            # Crop cloud mask to same geometry
            cloud_crop, _ = mask(cloud_src, gdf.geometry, crop=True, filled=True, nodata=1)

        # Apply cloud mask
        masked_data = np.full_like(pred_crop[0], fill_value=nodata_value)
        valid_mask = (cloud_crop[0] == 0)
        masked_data[valid_mask] = pred_crop[0][valid_mask]

        # Update profile for cropped data
        profile.update({
            'transform': pred_transform,
            'width': masked_data.shape[1],
            'height': masked_data.shape[0],
            'nodata': nodata_value,
        })

        # Write masked data
        with rasterio.open(output_file, 'w', **profile) as dst:
            if colormap:
                dst.write_colormap(1, colormap)
            if band_tags:
                dst.update_tags(1, **band_tags)
            dst.write(masked_data, 1)

    def _reproject_data(self, input_file: str, output_file: str) -> str:
        """
        Reproject raster data to project mask layer CRS

        :param input_file: input file path
        :param output_file: output file path
        :return: output file path
        """
        # Read mask file and get target CRS
        gdf = gpd.read_file(self.mask_file, layer=self.mask_layer)
        target_crs = gdf.crs.to_string()

        with rasterio.open(input_file) as src:
            src_crs = src.crs
            nodata_value = src.nodata if src.nodata is not None else 255
            band_tags = src.tags(1)

            src_transform = src.transform
            left, bottom, right, top = src.bounds


        # Check if CRS are the same
        if src_crs.to_string() == target_crs:
            logger.debug("Source and target CRS are identical - no reprojection needed")
            # Just copy the file if CRS are the same
            shutil.copy2(input_file, output_file)
            return output_file

        # Reproject to target CRS (mask layer CRS)
        src_pixel_size = abs(src_transform.a)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, target_crs, src.width, src.height, left, bottom, right, top, resolution=src_pixel_size,
        )

        # Calculate destination bounds for comparison
        dst_left = dst_transform.c
        dst_top = dst_transform.f
        dst_right = dst_left + dst_width * dst_transform.a
        dst_bottom = dst_top + dst_height * dst_transform.e

        warp_options = gdal.WarpOptions(
            dstSRS=target_crs,
            srcSRS=src_crs.to_string(),
            srcNodata=nodata_value,
            dstNodata=nodata_value,
            resampleAlg='near',
            format='GTiff',
            creationOptions=['COMPRESS=ZSTD', 'TILED=YES'],
            outputBounds=[dst_left, dst_bottom, dst_right, dst_top],  # Exact bounds
            xRes=src_pixel_size,
            yRes=src_pixel_size,
            targetAlignedPixels=True,
        )

        res = gdal.Warp(output_file, input_file, options=warp_options)
        res = None

        with rasterio.open(output_file, "r+") as dst:
            if band_tags:
                dst.update_tags(1, **band_tags)

        return output_file


    def _s3_to_http(self, url):
        """
        Convert s3 protocol to http protocol

        :param url: URL starts with s3://
        :return http url
        """
        if url.startswith('s3://'):
            s3_path = url
            bucket = s3_path[5:].split('/')[0]
            # NOTE: This is a workaround to earth search issue where the data points to a different bucket.
            # See issue: https://github.com/Element84/earth-search/issues/3
            if bucket != 'sentinel-s2-l1c':
                bucket = 'sentinel-s2-l1c'
            object_name = '/'.join(s3_path[5:].split('/')[1:])
            return 'https://{0}.s3.amazonaws.com/{1}'.format(bucket, object_name)
        else:
            return url


    def create(self, progress=None):
        logger.info(f'Entering create')
        t1 = time.time()
        predict_task = None
        landuse_prediction = LandusePrediction(item=self.item)
        if progress:
            predict_task = progress.add_task(f"[cyan]Starting {landuse_prediction.component_name} prediction")


        temp_file = f"{self.predicted_file}.jano"
        img_paths = [self.asset_files[band] for band in landuse_prediction.target_bands]
        col_size, row_size, min_resolution_path = landuse_prediction.preinference_check(img_paths)
        print(col_size, row_size, min_resolution_path)
        return temp_file

def test_jins_model(work_dir='/tmp'):
    """
    Testing the prediction speed/time as Jin designed it.
    :return:
    """
    item_url = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c/items/S2C_36NYG_20250219_0_L1C"
    mask = '/data/surge/ugaken.fgb'
    item = pystac.Item.from_file(item_url)

    with Progress() as progress:
        start_time = time.time()
        sentinel_item = SentinelItem(item, mask_file=None, mask_layer=None)
        sentinel_item.download_assets(download_dir=work_dir,
                                      progress=progress, force=False)
        down_time = time.time()
        logger.info(f"download time: {down_time - start_time}")

        downloaded_files = list(sentinel_item.asset_files.values())
        logger.info(f"downloaded files: {len(downloaded_files)} {downloaded_files}")
        #
        # sentinel_item.detect_cloud(progress=progress, force=True)
        #
        # t3 = time.time()
        # logger.info(f"cloud detection time: {t3 - t2}")
        #
        #temp_prediction_file = sentinel_item.predict(progress=progress)
        temp_prediction_file = sentinel_item.create(progress=progress)

        pred_time = time.time()
        logger.info(f"landuse prediction time: {pred_time-down_time} {os.path.abspath(temp_prediction_file)}")
        #
        # # temp_prediction_file = f"{sentinel_item.predicted_file}.tmp"
        # sentinel_item.mask_cloud_pixels(temp_prediction_file, sentinel_item.predicted_file, progress=progress)
        #
        # t5 = time.time()
        # logger.info(f"cloud masking time: {t5 - t4}, total time: {t5 - t1}")


if __name__ == '__main__':

    # import logging
    # logging.basicConfig()
    # logger = logging.getLogger()
    from rapida.util.setup_logger import setup_logger
    logger = setup_logger(name='rapida', make_root=False)
    test_jins_model()

    # item_url = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c/items/S2A_36MTC_20240819_0_L1C"
    # # item_url = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c/items/S2B_35MRU_20250421_0_L1C"
    # # item_url = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c/items/S2A_35MRT_20240819_0_L1C"
    # # item_url = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c/items/S2B_35MRT_20240715_0_L1C"
    # mask_file = "/data/rwanda_tests/data/rwanda_tests.gpkg"
    # mask_layer = "polygons"
    # download_dir = "/data/sentinel_tests"
    #
    # item = pystac.Item.from_file(item_url)
    #
    # with Progress() as progress:
    #     t1 = time.time()
    #     sentinel_item = SentinelItem(item, mask_file=mask_file, mask_layer=mask_layer)
    #     sentinel_item.download_assets(download_dir=download_dir,
    #                                   progress=progress)
    #     t2 = time.time()
    #     logger.info(f"download time: {t2 - t1}")
    #
    #     downloaded_files = list(sentinel_item.asset_files.values())
    #     logger.info(f"downloaded files: {downloaded_files}")
    #
    #     sentinel_item.detect_cloud(progress=progress, force=True)
    #
    #     t3 = time.time()
    #     logger.info(f"cloud detection time: {t3 - t2}")
    #
    #     temp_prediction_file = sentinel_item.predict(progress=progress)
    #
    #     t4 = time.time()
    #     logger.info(f"landuse prediction time: {t4 - t3}")
    #
    #     # temp_prediction_file = f"{sentinel_item.predicted_file}.tmp"
    #     sentinel_item.mask_cloud_pixels(temp_prediction_file, sentinel_item.predicted_file, progress=progress)
    #
    #     t5 = time.time()
    #     logger.info(f"cloud masking time: {t5 - t4}, total time: {t5 - t1}")

