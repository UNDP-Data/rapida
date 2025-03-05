import os
import logging
from typing import List
from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.session import Session
from cbsurge.util.resolve_url import resolve_geohub_url
from cbsurge.util.download_geodata import download_vector
from cbsurge.util.proj_are_equal import proj_are_equal
from cbsurge.project import Project
from osgeo import gdal, osr
import pyogrio
import shapely


logger = logging.getLogger(__name__)


class RoadsComponent(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def __call__(self, variables: List[str] = None, **kwargs) -> str:

        logger.debug(f'Assessing component "{self.component_name}" ')
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if not var_name in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        with Session() as ses:
            variables_data = ses.get_component(self.component_name)

            for var_name in variables:
                var_data = variables_data[var_name]
                var_data['source'] = resolve_geohub_url(var_data['source'], link_name="flatgeobuf")

                # create instance
                v = RoadsVariable(name=var_name, component=self.component_name, **var_data)
                # assess
                v(**kwargs)


class RoadsVariable(Variable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        project = Project(path=os.getcwd())
        self.local_path = project.geopackage_file_path

    def download(self, **kwargs):
        project = Project(path=os.getcwd())
        self.local_path = project.geopackage_file_path

        source = self.source

        force_compute = kwargs.get('force_compute', False)
        progress = kwargs.get('progress', False)

        layers = pyogrio.list_layers(self.local_path)
        layer_names = layers[:, 0]

        if force_compute == True or not self.component in layer_names:
            with gdal.OpenEx(project.geopackage_file_path, gdal.OF_READONLY | gdal.OF_VECTOR) as poly_ds:
                lyr = poly_ds.GetLayerByName(project.polygons_layer_name)
                mask_polygons = dict()
                for feat in lyr:
                    polygon = feat.GetGeometryRef()

                    try:
                        src_dataset_info = pyogrio.read_info(source)
                    except Exception as e:
                        if '404' in str(e):
                            logger.error(f'{source} is not reachable')
                        raise
                    src_crs = src_dataset_info['crs']
                    src_srs = osr.SpatialReference()
                    src_srs.SetFromUserInput(src_crs)
                    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

                    target_srs = lyr.GetSpatialRef()
                    will_reproject = not proj_are_equal(src_srs=src_srs, dst_srs=target_srs)
                    if will_reproject:
                        target2src_tr = osr.CoordinateTransformation(target_srs, src_srs)
                        polygon.Transform(target2src_tr)
                    shapely_polygon = shapely.wkb.loads(bytes(polygon.ExportToIsoWkb()))
                    poly_id = f'line::{feat.GetFID()}'
                    mask_polygons[poly_id] = shapely_polygon

                download_vector(
                    src_dataset_url=self.source,
                    src_layer_name=0,
                    dst_dataset_path=self.local_path,
                    dst_layer_name=self.component,
                    dst_srs=target_srs,
                    mask_polygons=mask_polygons,
                    progress=progress,
                    overwrite_dst_layer=True
                )
                lyr.SetAttributeFilter(None)
                lyr.ResetReading()
        return self.local_path

    def compute(self, **kwargs):
        pass

    def resolve(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass