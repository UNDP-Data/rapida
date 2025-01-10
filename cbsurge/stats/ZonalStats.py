import logging
import os
from exactextract import exact_extract
from cbsurge.stats.GDALRasterSource import GDALRasterSource
from cbsurge.stats.OGRDataSource import OGRDataSource
import geopandas

logger = logging.getLogger(__name__)


class ZonalStats:
    def __init__(self, input_file: str, target_crs: str="ESRI:54009"):
        """
        constructor

        parameters:
            input_file: path to input file
            target_crs: target spatial reference in EPSG code. Default is ESRI:54009 (Mollweide projection https://epsg.io/54009).
        """
        self.input_file = input_file
        self.target_crs = target_crs
        self.gdf = None
        logger.debug(f"input_file: {self.input_file}, target_crs: {self.target_crs}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.input_file = None
        self.target_crs = None
        self.gdf = None


    def compute(self, rasters: list, operations: list = None, operation_cols: list = None):
        """
        compute zonal statistics with given raster file.

        Parameters:
            rasters: list of paths to raster files
            operations:
                list of operations to compute.
                See available operations at https://isciences.github.io/exactextract/operations.html
            operation_cols:
                list of column names for each operation. The number of columns must match the number of operations.
                If this option is not specified, raster file name is used as prefix of column names.
                For example, if aaa.tif is specified, column name will be 'aaa_sum'.
        Returns:
            geopandas dataframe with zonal statistics
        """
        logger.info(f"start computing zonal statistics for {rasters}")
        if operations is None or len(operations) == 0:
            raise RuntimeError(f"at least one operation must be specified")

        if operation_cols is not None and len(operation_cols) > 0:
            if len(operation_cols) != len(operations) * len(rasters):
                raise RuntimeError(
                    f"The number of operation columns must match the number of operations multiplied by the number of rasters.")


        with OGRDataSource(self.input_file) as cleaned_ds:
            cleaned_input_file = cleaned_ds.clean()

            logger.debug(f"Input file was cleaned as {cleaned_input_file}")
            # vector
            with OGRDataSource(filepath=cleaned_input_file, is_clear=True) as vector_ds:
                vector_fields = vector_ds.get_fields()

                reprojected_vector_path = vector_ds.reproject(
                    target_crs=self.target_crs,
                    data_format='FlatGeobuf')
                logger.debug(f"vector ({cleaned_input_file}) was reprojected to {self.target_crs} as {reprojected_vector_path}")

                combined_results = None
                for raster_index, raster in enumerate(rasters):
                    # raster
                    with GDALRasterSource(raster) as raster_ds:
                        reprojected_rast = raster_ds.reproject(self.target_crs)
                        logger.debug(f"raster ({raster}) was reprojected to {self.target_crs} as {reprojected_rast}")

                        gdf = exact_extract(
                            reprojected_rast,
                            reprojected_vector_path,
                            ops=operations,
                            include_geom=True,
                            output="pandas",
                            progress=True
                        )
                        raster_name_with_ext = os.path.basename(raster)
                        raster_file_name = os.path.splitext(raster_name_with_ext)[0]

                        for index, ope in enumerate(operations):
                            column_name = ope
                            if column_name in gdf.columns:
                                col_index = raster_index * len(operations) + index
                                new_col_name = f"{raster_file_name}_{column_name}"
                                if operation_cols is not None and len(operation_cols) > 0:
                                    new_col_name = operation_cols[col_index]
                                gdf.rename(columns={column_name: new_col_name}, inplace=True)

                        if combined_results is None:
                            combined_results = gdf
                        else:
                            combined_results = combined_results.merge(gdf, on=['geometry'], how='inner')

                        os.remove(reprojected_rast)
                        logger.debug(f"deleted {reprojected_rast}")
                combined_results = combined_results.merge(geopandas.read_file(reprojected_vector_path), on='geometry', how='inner')
                os.remove(reprojected_vector_path)
                logger.debug(f"deleted {reprojected_vector_path}")
                logger.info(f"end computing zonal statistics for {rasters}")
                self.gdf = combined_results
                return self.gdf

    def write(self, output_file, target_crs: str = "EPSG:3857"):
        """
        write zonal statistics to output file.
        compute method must be executed prior to calling this method.

        Parameters:
            output_file: path to output file. Only supports .shp, .gpkg, .fgb, .geojson currently.
            target_crs: target spatial resolution in EPSG code. Default is 3857
        """
        if self.gdf is None:
            raise RuntimeError(f"execute compute() method first to compute ZonalStats.")
        gdf_copy = self.gdf.copy()
        if target_crs:
            gdf_copy = gdf_copy.to_crs(self.target_crs)

        driver = None
        if output_file.endswith(".shp"):
            driver = "ESRI Shapefile"
        elif output_file.endswith(".gpkg"):
            driver = "GPKG"
        elif output_file.endswith(".fgb"):
            driver = "FlatGeobuf"
        elif output_file.endswith(".geojson"):
            driver = "GeoJSON"
        else:
            raise RuntimeError(f"unsupported output file type: {output_file}")
        gdf_copy.to_file(output_file, driver=driver)
        logger.info(f"Saved stats to {output_file}")
