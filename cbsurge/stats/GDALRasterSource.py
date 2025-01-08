import logging
import os
from osgeo import gdal


logger = logging.getLogger(__name__)


class GDALRasterSource:
    """
    The class manages raster data source and additional functionality to process.

    Example:
        with GDALRasterSource(input_raster) as raster:
            raster.reproject_raster(output_raster, target_srs)
    """

    def __init__(self, filepath: str, is_clear: bool = False):
        """
        Constructor.

        Parameters:
            filepath (str): Path to the vector file
            is_clear (bool): If True, the file will be deleted when exiting from with statement
        """
        self.filepath = filepath
        self.is_clear = is_clear
        self.datasource = None

    def __enter__(self):
        """Open the raster data source."""
        self.dataset = gdal.Open(self.filepath)
        if not self.dataset:
            raise RuntimeError(f"Failed to open raster file: {self.filepath}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the raster data source."""
        if self.dataset:
            self.dataset = None
        if self.is_clear:
            self.delete_file()

    def delete_file(self):
        """Delete file"""
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
            logger.debug(f"Deleted file: {self.filepath}")


    def reproject(self,
                  target_srs,
                  resample_alg=gdal.GRA_Bilinear,
                  data_format="GTiff"):
        """
        Reproject the raster file to a specified coordinate reference system.

        Parameters:
            target_srs (str): EPSG code or PROJ.4 string for the target projection.
            resample_alg (int): Resampling algorithm (default: gdal.GRA_Bilinear).
            data_format (str): Output raster format (default: "GTiff").
        """
        file_name_with_ext = os.path.basename(self.filepath)

        raster_file = os.path.splitext(file_name_with_ext)[0]
        reprojected_rast = f"{raster_file}_reprojected.tif"

        filepath, filename = os.path.split(self.filepath)

        reprojected_rast_path = os.path.join(filepath, reprojected_rast)
        warp_options = gdal.WarpOptions(
            dstSRS=target_srs,
            resampleAlg=resample_alg,
            format=data_format
        )

        result = gdal.Warp(
            destNameOrDestDS=reprojected_rast_path,
            srcDSOrSrcDSTab=self.filepath,
            options=warp_options
        )

        if result is None:
            raise RuntimeError("gdal.Warp failed to reproject the raster.")
        result = None
        logger.debug(f"Reprojected raster saved to {reprojected_rast}")
        return reprojected_rast_path
