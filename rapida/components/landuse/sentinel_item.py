import json
import logging
import os
import asyncio
import time
import re
import httpx
import pystac
from typing import Dict
from datetime import datetime
from osgeo import ogr, osr, gdal
import rasterio
import numpy as np
from rich.progress import Progress

from rapida.components.landuse.cloud import cloud_detect
from rapida.components.landuse.prediction import predict
from rapida.components.landuse.constants import SENTINEL2_ASSET_MAP
from rapida.util.setup_logger import setup_logger


logger = logging.getLogger(__name__)


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
        ID for the class instance. It uses grid:code for unique ID.
        """
        tile_id = self._item.properties.get("grid:code")
        if tile_id is None:
            tile_id = self._item.id
        return tile_id

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
            ds = ogr.Open(self.mask_file)
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

    def is_valid_item(self):
        """
        Validate whether this STAC item contains all required assets
        """
        return all(asset in self.item.assets for asset in self.target_asset)

    def download_assets(self,
                        download_dir: str,
                        progress=None,
                        max_workers: int = 12) -> Dict[str, str]:
        """
        Download all required bands for this STAC item in parallel.

        :param download_dir: directory to store downloaded GeoTIFFs
        :param progress: optional rich progress bar
        :param max_workers: maximum number of parallel downloads
        :return: Dictionary of band name -> downloaded file path
        """
        self._asset_files = {
            band_name: os.path.join(download_dir, self.id, f"{band_name}.tif")
            for asset_key, band_name in self.target_asset.items()
        }

        item_path = os.path.join(download_dir, self.id, "item.json")
        if os.path.exists(item_path):
            with open(item_path, "r", encoding="utf-8") as f:
                existing_item = json.load(f)
                if existing_item.get("id") == self.item.id:
                    logger.warning(f"The same item ({self.id}) has already been downloaded. Skipping")
                    return self.asset_files
                else:
                    # if different item id, delete old prediction file if exists
                    if os.path.exists(self.predicted_file):
                        os.remove(self.predicted_file)
                    if os.path.exists(self.cloud_mask_file):
                        os.remove(self.cloud_mask_file)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        semaphore = asyncio.Semaphore(max_workers)

        async def download_all():
            async def download_one(asset_key: str, band_name: str):
                asset = self.item.assets[asset_key]
                url = self._s3_to_http(asset.href)
                band_dir = os.path.join(download_dir, self.id)
                os.makedirs(band_dir, exist_ok=True)
                target_path = os.path.join(band_dir, band_name)

                max_retries = 5
                for attempt in range(1, max_retries + 1):
                    async with semaphore:
                        try:
                            result = await self._download_asset(
                                file_url=url,
                                target=target_path,
                                target_srs=self.target_srs,
                                no_data_value=self.asset_nodata[band_name],
                                progress=progress,
                            )
                            # self._asset_files[band_name] = result
                            return
                        except Exception as e:
                            logger.warning(f"[{band_name}] Attempt {attempt}/{max_retries} failed: {e}")
                            if attempt == max_retries:
                                logger.error(f"Failed to download {band_name} after {max_retries} attempts.")
                            else:
                                await asyncio.sleep(3)

            tasks = [
                asyncio.create_task(download_one(asset_key, band_name))
                for asset_key, band_name in self.target_asset.items()
            ]
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                logger.warning("Download interrupted by user. Cancelling tasks...")
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        loop.run_until_complete(download_all())
        loop.close()

        # once all downloads are successfully done, write item.json in the folder
        with open(item_path, "w", encoding="utf-8") as f:
            json.dump(self.item.to_dict(), f, indent=2)

        return self.asset_files


    def predict(self, progress=None)->str:
        """
        Predict land use from downloaded assets

        :return: output file path
        """
        land_use_target_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']
        temp_file = f"{self.predicted_file}.tmp"
        try:

            img_paths = [self.asset_files[band] for band in land_use_target_bands]
            predict(
                img_paths=img_paths,
                output_file_path=temp_file,
                # num_workers=1,
                progress=progress,
            )
            self._mask_cloud_pixels(temp_file, self.predicted_file)
            return self.predicted_file
        except Exception as e:
            # delete incomplete predicted file if error occurs
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(self.predicted_file):
                os.remove(self.predicted_file)
            raise e

    def detect_cloud(self, progress=None)->str:
        """
        Detect cloud from downloaded assets

        :return: output file path
        """
        cloud_target_bands = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
        try:
            img_paths = [self.asset_files[band] for band in cloud_target_bands]
            cloud_detect(
                img_paths=img_paths,
                output_file_path=self.cloud_mask_file,
                # num_workers=1,
                progress=progress,
            )
            return self.cloud_mask_file
        except Exception as e:
            # delete incomplete predicted file if error occurs
            if os.path.exists(self.cloud_mask_file):
                os.remove(self.cloud_mask_file)
            raise e

    def _mask_cloud_pixels(self, input_file: str, output_file: str)-> str:
        """
        Mask cloud pixels from land use prediction image

        :param input_file: input file path of land use prediction image
        :param output_file: output file path of mask prediction image
        :return: output file path of mask prediction image
        """
        if not os.path.exists(self.cloud_mask_file):
            os.rename(input_file, output_file)
            return output_file

        with rasterio.open(input_file) as pred_src, \
                rasterio.open(self.cloud_mask_file) as cloud_src:
            pred_data = pred_src.read(1)
            cloud_mask = cloud_src.read(1)

            nodata_value = pred_src.nodata if pred_src.nodata is not None else 255
            band_count = pred_src.count

            # Apply mask
            masked_data = np.where(cloud_mask == 1, nodata_value, pred_data)
            # Copy metadata profile
            profile = pred_src.profile.copy()
            profile.update({
                'nodata': nodata_value,
                'count': band_count,
            })
            profile.pop('photometric', None)

            # Write masked output
            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(masked_data, 1)

                for b in range(1, band_count + 1):
                    # copy band colormap
                    colormap = pred_src.colormap(b)
                    dst.write_colormap(b, colormap)

                    # Copy band-metadata
                    band_tags = pred_src.tags(b)
                    if band_tags:
                        dst.update_tags(b, **band_tags)

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
            object_name = '/'.join(s3_path[5:].split('/')[1:])
            return 'https://{0}.s3.amazonaws.com/{1}'.format(bucket, object_name)
        else:
            return url


    def _harmonize_to_old(self, data, nodata_value):
        """
        Harmonize new Sentinel-2 data to the old baseline by subtracting a fixed offset.
        described at https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change

        :param data: Sentinel-2 data to harmonize
        :param nodata_value: nodata value to exclude from harmonization
        :return: harmonized data. The input data with an offset of 1000 subtracted.
        """
        offset = 1000
        harmonized = np.where(data != nodata_value, data - offset, data)
        return harmonized


    async def _download_asset(
            self,
            file_url: str,
            target: str,
            target_srs: str,
            no_data_value: int = 0,
            progress=None,
    ) -> str:
        """
        Download JP2 file from STAC server, then reproject and crop by project area, and transform to GeoTiff file.

        :param file_url: file URL to download
        :param target: target file name without extension
        :param no_data_value: nodata value. default is 0
        :param progress: rich progress instance
        :return: output file URL
        """
        download_task = None
        if progress is not None:
            download_task = progress.add_task(
                description=f'[blue] Downloading {file_url}', total=None)

        extension = os.path.splitext(file_url)[1]
        download_file = f"{target}.tif"

        jp2_file = f"{target}{extension}"

        try:
            # fetch content-length from remote file header
            async with httpx.AsyncClient() as client:
                head_resp = await client.head(file_url)
                head_resp.raise_for_status()
                remote_content_length = head_resp.headers.get("content-length")
                if remote_content_length is None:
                    raise ValueError("No content-length in response headers")
                remote_content_length = int(remote_content_length)

            if os.path.exists(download_file):
                with rasterio.open(download_file) as src:
                    meta_content_length = src.tags().get("JP2_CONTENT_LENGTH")
                    if meta_content_length and int(meta_content_length) == remote_content_length:
                        logging.debug(f"file already exists. Skipped: {download_file}")
                        return download_file

            pattern = r"/(\d{4})/(\d{1,2})/(\d{1,2})/"
            match = re.search(pattern, file_url)
            if match:
                year, month, day = map(int, match.groups())
                acquisition_date = datetime(year, month, day)
                logging.debug("Extracted acquisition date: %s", acquisition_date)
            else:
                logging.error("Failed to extract date from file_url: %s", file_url)
                raise ValueError(f"Could not extract date from URL: {file_url}")

            cutoff = datetime(2022, 1, 25)

            async with httpx.AsyncClient() as client:
                async with client.stream("GET", file_url) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))

                    if progress is not None and download_task is not None:
                        progress.update(download_task, total=total)

                    with open(jp2_file, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            if progress is not None and download_task is not None:
                                progress.update(download_task, advance=len(chunk))

            if progress and download_task:
                progress.update(download_task, description=f"[blue] Reprojecting {jp2_file}...")

            with rasterio.open(jp2_file) as src:
                data = src.read()

                nodata = src.nodata if src.nodata is not None else no_data_value

                if acquisition_date >= cutoff:
                    data = self._harmonize_to_old(data, nodata)  # Ensure harmonize_to_old is defined

                profile = src.profile
                profile.update(driver="GTiff", dtype=data.dtype, count=src.count)

                vsimem_path = "/vsimem/in.tif"
                with rasterio.MemoryFile() as memfile:
                    with memfile.open(**profile) as dataset:
                        dataset.write(data)
                    gdal.FileFromMemBuffer(vsimem_path, memfile.read())

                xRes = yRes = self.min_resolution

                warp_options = gdal.WarpOptions(
                    dstSRS=target_srs.ExportToWkt(),
                    format="GTiff",
                    xRes=xRes,
                    yRes=yRes,
                    creationOptions=["COMPRESS=ZSTD", "TILED=YES"],
                    srcNodata=nodata,
                    dstNodata=nodata,
                    resampleAlg="near",
                    cutlineDSName=self.mask_file,
                    cutlineLayer=self.mask_layer,
                    cropToCutline=True,
                )

                rds = gdal.Warp(destNameOrDestDS=download_file,
                                srcDSOrSrcDSTab=vsimem_path,
                                options=warp_options)
                metadata = {"JP2_CONTENT_LENGTH": str(remote_content_length)}
                item_datetime = self.item.datetime.isoformat() if self.item.datetime else None
                if item_datetime is not None:
                    metadata["ITEM_DATETIME"] = item_datetime
                cloud_cover = self.item.properties.get("eo:cloud_cover")
                if cloud_cover is not None:
                    metadata["CLOUD_COVER"] = str(cloud_cover)
                rds.SetMetadata(metadata)
                gdal.Unlink(vsimem_path)
                rds = None
        finally:
            if os.path.exists(jp2_file):
                os.remove(jp2_file)
            if progress and download_task:
                progress.update(download_task, description=f"[blue] Downloaded file saved to {download_file}")
                progress.remove_task(download_task)

        return download_file


if __name__ == '__main__':
    setup_logger()

    item_url = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c/items/S2B_35MRT_20240715_0_L1C"
    mask_file = "/data/kigali/data/kigali.gpkg"
    mask_layer = "polygons"
    download_dir = "/data/sentinel_tests"

    item = pystac.Item.from_file(item_url)

    with Progress() as progress:
        t1 = time.time()
        sentinel_item = SentinelItem(item, mask_file=mask_file, mask_layer=mask_layer)
        sentinel_item.download_assets(download_dir=download_dir,
                                      progress=progress)
        t2 = time.time()
        logger.info(f"download time: {t2 - t1}")

        downloaded_files = list(sentinel_item.asset_files.values())
        logger.info(f"downloaded files: {downloaded_files}")

        sentinel_item.predict(progress=progress)

        t3 = time.time()
        logger.info(f"landuse prediction time: {t3 - t2}, total time: {t3 - t1}")

        sentinel_item.detect_cloud(progress=progress)

        t4 = time.time()
        logger.info(f"cloud detection time: {t4 - t3}, total time: {t4 - t1}")