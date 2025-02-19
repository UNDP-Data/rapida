import logging
import os
import geopandas as gpd
from osgeo import ogr, gdal


logger = logging.getLogger(__name__)


class OGRDataSource:
    """
    The class manages vector data source from OGR API

    Example:
        with OGRDataSource(filepath) as datasource:
            datasource.get_fields()
    """
    def __init__(self, filepath: str, is_clear:bool=False):
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
        self.datasource = ogr.Open(self.filepath)
        if not self.datasource:
            raise RuntimeError(f"Failed to open vector file: {self.filepath}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.datasource:
            self.datasource = None
        if self.is_clear:
            self.delete_file()

    def delete_file(self):
        """Delete file"""
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
            logger.debug(f"Deleted file {self.filepath}")


    def get_layer(self):
        """Get the layer from the vector data source."""
        return self.datasource.GetLayer()

    def clean(self):
        """
        clean dataset before doing zonal stats
        """
        gdf = gpd.read_file(self.filepath)

        for column in gdf.columns:
            dtype = gdf[column].dtype
            if dtype == 'object':
                gdf[column] = gdf[column].astype(str)
            elif dtype == 'int64':
                gdf[column] = gdf[column].astype('float64')

        dir_name = os.path.dirname(self.filepath)
        file, ext = self.split_filename(self.filepath)
        output_file = f"{dir_name}/{file}_cleaned{ext}"

        gdf.to_file(output_file, driver="GeoJSON")

        return output_file

    def get_fields(self):
        """Get a list of field names from the vector dataset."""
        layer = self.get_layer()
        if not layer:
            raise RuntimeError(f"Failed to access layer in vector file: {self.filepath}")

        layer_def = layer.GetLayerDefn()
        field_names = [layer_def.GetFieldDefn(i).GetName() for i in range(layer_def.GetFieldCount())]
        return field_names

    def reproject(self,
                  target_crs,
                  src_crs=None,
                  data_format="FlatGeobuf",
                  output_file=None):
        """
        Reproject the vector file to FlatGeobuf format with a specified coordinate reference system.

        Parameters:
            target_crs (str): EPSG code for the target projection. Example: "EPSG:3857"
            src_crs (str): EPSG code for the source projection (optional). Example: "EPSG:4326"
            data_format (str): The format to use (defaults to "FlatGeobuf").
            output_file (str): Optional. Path to save the reprojected vector file if specified. Otherwise, it creates a new file with _reprojected suffix.
        """
        if not output_file:
            vector_file = self.split_filename(self.filepath)[0]
            output_file = f"{vector_file}_reprojected.fgb"

        translate_options = gdal.VectorTranslateOptions(
            format=data_format,
            dstSRS=target_crs,
            srcSRS=src_crs,
            layerName=self.split_filename(output_file)[0]
        )

        gdal.VectorTranslate(
            destNameOrDestDS=output_file,
            srcDS=self.filepath,
            options=translate_options,
        )

        logger.debug(f"Reprojected vector saved to {output_file} in {format} format")
        return output_file

    def split_filename(self, filename):
        """
        Split file name from extension

        Returns:
            It returns an array which contains filename without extension, extension
        """
        file_name_with_ext = os.path.basename(filename)
        file_name_without_ext = os.path.splitext(file_name_with_ext)[0]
        ext = os.path.splitext(file_name_with_ext)[1]
        return [file_name_without_ext, ext]