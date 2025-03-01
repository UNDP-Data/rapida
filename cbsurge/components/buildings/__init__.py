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

from cbsurge.util.proj_are_equal import proj_are_equal
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

    def download(self, **kwargs):
        logger.debug(f'Downloading {self.name}')
        project = Project(os.getcwd())
        progress = kwargs.get('progress', None)

        # if os.path.exists(self.local_path):
        #     assert os.path.exists(self.affected_path), f'{self.affected_path} does not exist'
        #     return self.local_path
        sources = list()
        with gdal.OpenEx(project.geopackage_file_path, gdal.OF_READONLY | gdal.OF_VECTOR) as poly_ds:
            lyr = poly_ds.GetLayerByName(project.polygons_layer_name)
            for country in project.countries:
                logger.debug(f'Downloading {self.name} in {country}')
                source = self.interpolate_template(template=self.source, country=country, **kwargs)
                source, layer_name = source.split('::')
                src_dataset_info = pyogrio.read_info(source, layer_name)
                src_crs = src_dataset_info['crs']
                src_srs = osr.SpatialReference()
                src_srs.SetFromUserInput(src_crs)
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                lyr.SetAttributeFilter(f'iso3="{layer_name}"')
                target_srs = lyr.GetSpatialRef()
                will_reproject = not proj_are_equal(src_srs=src_srs, dst_srs=target_srs)
                if will_reproject:
                    target2src_tr = osr.CoordinateTransformation(target_srs, src_srs)
                mask_polgygons = dict()
                for feat in lyr:
                    polygon = feat.GetGeometryRef()
                    if will_reproject:
                        polygon.Transform(target2src_tr)
                    shapely_polygon = shapely.wkb.loads(bytes(polygon.ExportToIsoWkb()))
                    poly_id = feat['name'] or feat.GetFID()
                    mask_polgygons[poly_id] = shapely_polygon

                download_vector(
                    src_dataset_url=source,
                    src_layer_name=layer_name,
                    dst_dataset_path=project.geopackage_file_path,
                    dst_layer_name=self.component, dst_srs=target_srs,
                    mask_polygons=mask_polgygons,
                    progress=progress
                )
                lyr.SetAttributeFilter(None)
                lyr.ResetReading()
    def compute(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass
    def resolve(self, **kwargs):
        pass