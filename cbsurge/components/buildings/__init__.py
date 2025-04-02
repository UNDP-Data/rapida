import pandas as pd
from shapely.creation import polygons

from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.util.download_geodata import download_vector
from cbsurge.project import  Project
from cbsurge.session import Session
import os
import logging
import pyogrio
from osgeo import gdal, osr
import shapely
from cbsurge.components.buildings.preprocessing import mask_buildings
from cbsurge.util.proj_are_equal import proj_are_equal
import geopandas as gpd

logger = logging.getLogger(__name__)




# import click
# from cbsurge.components.builtenv.buildings.fgb import download_bbox, download_admin
# from cbsurge.components.builtenv.buildings.pmt import download as download_pmt, GMOSM_BUILDINGS
# from cbsurge.util.bbox_param_type import BboxParamType
# import asyncio
#
# @click.group()
# def buildings():
#     f"""Command line interface for {__package__} package"""
#     pass
#
# @click.group()
# def download():
#     f"""Command line interface for {__package__} package"""
#     pass
#
#
# @download.command(no_args_is_help=True)
#
# @click.option('-b', '--bbox', required=True, type=BboxParamType(),
#               help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north' )
# @click.option('-o', '--out-path', required=True, type=click.Path(),
#               help='Full path to the buildings dataset' )
#
# @click.option('-bs', '--batch-size', type=int, default=65535,
#               help='The max number of buildings to be downloaded in one chunk or batch. '
#                    )
# def fgbbbox(bbox=None, out_path=None, batch_size:[int,None]=1000):
#     """
#         Download/stream buildings from VIDA buildings using pyogrio/pyarrow API
#
#         :param bbox: iterable of floats, xmin, ymin, xmax,ymax
#         :param out_path: str, full path where the buildings layer will be written
#         :param batch_size: int, default=1000, the max number of buildings to download in one batch
#         If supplied, the buildings are downloaded in batches otherwise they are streamd through pyarrow library
#
#     """
#     download_bbox(bbox=bbox, out_path=out_path, batch_size=batch_size)
#
#
#
# @download.command(no_args_is_help=True)
# @click.option('-a', '--admin-path', required=True, type=click.Path(),
#               help='Full path to the admin dataset' )
# @click.option('-o', '--out-path', required=True, type=click.Path(),
#               help='Full path to the buildings dataset' )
# @click.option('--country-col-name', required=True, type=str,
#               help='The name of the column from the admin layer attributes that contains the ISO3 country code' )
# @click.option('--admin-col-name', required=True, type=str,
#               help='The name of the column from the admin layer attributes that contains the admin unit name' )
# @click.option('-bs', '--batch-size', type=int, default=65535,
#               help='The max number of buildings to be dowloaded in one chunk or batch. ')
#
# def fgbadmin(admin_path=None, out_path=None, country_col_name=None, admin_col_name=None, batch_size=None):
#
#     """Fetch buildings from VIDA FGB based on the boundaries of the admin units"""
#
#
#     download_admin(admin_path=admin_path,out_path=out_path,
#                    country_col_name=country_col_name, admin_col_name=admin_col_name, batch_size=batch_size)
#
#
#
# @download.command(no_args_is_help=True)
#
# @click.option('-b', '--bbox', required=True, type=BboxParamType(),
#               help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north' )
# @click.option('-o', '--out-path', required=True, type=click.Path(),
#               help='Full path to the buildings dataset' )
#
# @click.option('-z', '--zoom-level', type=int,
#               help='The zoom level from which to fetch the buildings.Defaults to dataset max zoom-1')
# @click.option('-x', type=int,
#               help='The x coordinate of the til;e for a specific zoom level')
# @click.option('-y', type=int,
#               help='The y coordinate of the til;e for a specific zoom level')
#
# def pmt(bbox=None, out_path=None, zoom_level=None,  x=None,y=None):
#     """
#
#         Fetch buildings from url remote source in PMTiles format from VIDA
#         https://data.source.coop/vida/google-microsoft-osm-open-buildings/pmtiles/goog_msft_osm.pmtiles
#
#         :param bbox: iterable of floats, xmin, ymin, xmax,ymax
#         :param out_path: str, full path where the buildings layer will be written
#         :param zoom_level: int the zoom level, defaults to max zoom level -1
#         :param x:int,  the tile x coordinate at a specific zoom level, can be used as a filter
#         :param y: int, the tile y coordinate at a specific zoom level, can be sued as a filter
#
#     """
#     asyncio.run(download_pmt(bbox=bbox,out_path=out_path, zoom_level=zoom_level, x=x, y=y))
#
# buildings.add_command(download)
#
#


class BuildingsComponent(Component):

    def __call__(self, variables=None, **kwargs):
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if not var_name in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        project = Project(path=os.getcwd())
        logger.info(f'Assessing component "{self.component_name}" in  {", ".join(project.countries)}')
        with Session() as ses:
            variables_data = ses.get_component(self.component_name)

            for var_name in variables:
                var_data = variables_data[var_name]
                # create instance
                v = BuildingsVariable(
                    name=var_name,
                    component=self.component_name,
                    **var_data
                )


                # assess
                v(**kwargs)

class BuildingsVariable(Variable):

    def __call__(self, *args, **kwargs):
        """
                Assess a variable. Essentially this means a series of steps in a specific order:
                    - download
                    - preprocess
                    - analysis/zonal stats

                :param kwargs:
                :return:
                """

        force_compute = kwargs.get('force_compute', False)
        progress = kwargs.get('progress', None)

        if progress is not None:
            variable_task = progress.add_task(
                description=f'[blue]Assessing {self.component}->{self.name}', total=None)

        if not self.dep_vars:  # simple variable,
            if not force_compute:
                # logger.debug(f'Downloading {self.name} source')
                self.download(**kwargs)
                if progress is not None and variable_task is not None:
                    progress.update(variable_task, description=f'[blue]Downloaded {self.component}->{self.name}')
            else:
                # logger.debug(f'Computing {self.name} using gdal_calc from sources')
                self.compute(**kwargs)
                if progress is not None and variable_task is not None:
                    progress.update(variable_task, description=f'[blue]Computed {self.component}->{self.name}', )

        else:
            if self.operator:
                if not force_compute:
                    # logger.debug(f'Downloading {self.name} from  source')
                    self.download(**kwargs)
                    if progress is not None and variable_task is not None:
                        progress.update(variable_task, description=f'[blue]Downloaded {self.component}->{self.name}')
                else:
                    # logger.info(f'Computing {self.name}={self.sources} using GDAL')
                    self.compute(**kwargs)
                    if progress is not None and variable_task is not None:
                        progress.update(variable_task, description=f'[blue]Computed {self.component}->{self.name}')
            else:
                # logger.debug(f'Computing {self.name}={self.sources} using GeoPandas')
                sources = self.resolve(**kwargs)

        self.evaluate(**kwargs)
        if progress is not None and variable_task is not None:
            progress._tasks[variable_task].total = 1
            progress.update(variable_task, description=f'[blue]Assessed {self.component}->{self.name}', advance=1)

    def download(self, progress=None, force_compute=None, **kwargs):

        logger.debug(f'Downloading {self.name}')
        project = Project(os.getcwd())
        dst_layers = pyogrio.list_layers(project.geopackage_file_path)
        if self.component in dst_layers[:,0]  and not force_compute:
            if 'mask' in dst_layers:
                affected_layer_name = f'{self.component}.affected'
                assert affected_layer_name in dst_layers, f'layer {affected_layer_name}  does not exist'
            self.local_path = f'{project.geopackage_file_path}::{self.component}'
            return
        with gdal.OpenEx(project.geopackage_file_path, gdal.OF_READONLY | gdal.OF_VECTOR) as poly_ds:
            lyr = poly_ds.GetLayerByName(project.polygons_layer_name)
            for ci, country in enumerate(project.countries):
                mode = 'w' if ci == 0 else 'a'
                logger.info(f'Downloading {self.name} in {country}')
                source = self.interpolate_template(template=self.source, country=country, **kwargs)
                source, layer_name = source.split('::')
                try:
                    src_dataset_info = pyogrio.read_info(source, layer_name)
                except Exception as e:
                    logger.error(f'Failed to fetch info on buildings in {country}')
                    if '404' in str(e):
                        logger.error(f'{source} is not reachable')
                    raise
                src_crs = src_dataset_info['crs']
                src_srs = osr.SpatialReference()
                src_srs.SetFromUserInput(src_crs)
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                lyr.SetAttributeFilter(f'iso3="{layer_name}"')
                target_srs = lyr.GetSpatialRef()
                will_reproject = not proj_are_equal(src_srs=src_srs, dst_srs=target_srs)
                if will_reproject:
                    target2src_tr = osr.CoordinateTransformation(target_srs, src_srs)
                mask_polygons = dict()
                for feat in lyr:
                    polygon = feat.GetGeometryRef()
                    if will_reproject:
                        polygon.Transform(target2src_tr)
                    shapely_polygon = shapely.wkb.loads(bytes(polygon.ExportToIsoWkb()))
                    poly_id = feat['h3id'] or feat.GetFID()
                    mask_polygons[poly_id] = shapely_polygon

                download_vector(
                    src_dataset_url=source,
                    src_layer_name=layer_name,
                    dst_dataset_path=project.geopackage_file_path,
                    dst_layer_name=self.component,
                    dst_srs=target_srs,
                    dst_layer_mode=mode,
                    mask_polygons=mask_polygons,
                    progress=progress,
                    add_polyid=True

                )
                lyr.SetAttributeFilter(None)
                lyr.ResetReading()
        self.local_path = f'{project.geopackage_file_path}::{self.component}'
        self._compute_affected_(progress=progress )
        return self.local_path

    def _compute_affected_(self,  progress=None):

        project = Project(os.getcwd())
        if project.raster_mask is not None:

            logger.info('Calculating affected buildings')
            affected_layer_name = f'{self.component}.affected'
            mask_buildings(
                buildings_dataset=project.geopackage_file_path,
                buildings_layer_name=self.component,
                mask_ds_path=project.raster_mask,
                masked_buildings_dataset=project.geopackage_file_path,
                masked_buildings_layer_name=affected_layer_name,
                horizontal_chunks=10,
                vertical_chunks=10,
                workers=4,
                progress=progress,


            )


    def compute(self, force_compute=True, **kwargs):
        assert force_compute, f'invalid force_compute={force_compute}'
        return self.download(force_compute=force_compute, **kwargs)

    def evaluate(self, **kwargs):
        destination_layer = f'stats.{self.component}'
        affected_var_name = f'{self.name}_affected'
        affected_var_percentage_name = f'{affected_var_name}_percentage'
        project = Project(os.getcwd())
        logger.info(f'Evaluating variable {self.name}')
        dataset_path, layer_name = self.local_path.split('::')
        buildings_gdf = gpd.read_file(filename=dataset_path,layer=layer_name)
        layers = pyogrio.list_layers(dataset_path)
        layer_names = layers[:, 0]

        if destination_layer in layer_names:
            polygons_layer = destination_layer
        else:
            polygons_layer = project.polygons_layer_name

        polygons_gdf = gpd.read_file(project.geopackage_file_path,layer=polygons_layer)
        polygons_gdf = polygons_gdf.rename(columns={'h3id': 'polyid'})

        if self.name == 'nbuildings':
            var_gdf  = buildings_gdf.groupby('polyid').size().reset_index(name=self.name)
        else:
            buildings_gdf[self.name] = buildings_gdf.geometry.area
            var_gdf = buildings_gdf.groupby('polyid')[self.name].sum().reset_index()
        if project.raster_mask is not None:
            affected_layer_name = f'{self.component}.affected'
            if affected_layer_name in layer_names:
                affected_buildings_gdf = gpd.read_file(filename=dataset_path, layer=affected_layer_name)
                if self.name == 'nbuildings':
                    affected_var_gdf  = affected_buildings_gdf.groupby('polyid').size().reset_index(name=affected_var_name)
                else:
                    affected_buildings_gdf[affected_var_name] = affected_buildings_gdf.geometry.area
                    affected_var_gdf = affected_buildings_gdf.groupby('polyid')[affected_var_name].sum().reset_index()
                var_gdf = var_gdf.merge(affected_var_gdf, on='polyid', how='inner')
                var_gdf.eval(f"{affected_var_percentage_name}={f'{affected_var_name}'}/{self.name}*100", inplace=True)

        for col in [self.name, affected_var_name, affected_var_percentage_name]:
            if col in polygons_gdf.columns:
                polygons_gdf.drop(columns=[col], inplace=True)


        out_gdf = polygons_gdf.merge(var_gdf, on='polyid', how='inner')
        out_gdf = out_gdf.rename(columns={'polyid':'h3id'})
        out_gdf.to_file(dataset_path,layer=destination_layer,driver='GPKG', mode='w')



    def resolve(self, **kwargs):
        pass