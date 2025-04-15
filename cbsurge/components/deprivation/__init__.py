import os
import logging
from typing import List
from osgeo import gdal
from cbsurge.core.component import Component
from cbsurge.components.rwi import RwiVariable
from cbsurge.project.project import Project
from cbsurge.session import Session
from cbsurge.util import geo
from cbsurge.util.download_geodata import download_raster
from cbsurge.util.resolve_url import resolve_geohub_url
from osgeo_utils.gdal_calc import Calc
from cbsurge.constants import GTIFF_CREATION_OPTIONS

logger = logging.getLogger(__name__)


class DeprivationComponent(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def __call__(self, variables: List[str] = None, **kwargs) -> str:

        logger.info(f'Assessing component "{self.component_name}" ')
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if not var_name in self.variables:
                    msg = f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"'
                    logger.error(msg)
                    return
        multiple_vars = len(variables) > 1
        with Session() as ses:
            variables_data = ses.get_component(self.component_name)

            for var_index, var_name in enumerate(variables):
                var_data = variables_data[var_name]
                var_data['source'] = resolve_geohub_url(var_data['source'])

                # create instance
                v = DeprivationVariable(name=var_name, component=self.component_name, **var_data)
                # assess
                if multiple_vars and kwargs['force'] is True:
                    if var_index > 0:
                        kwargs['force'] = False

                v(**kwargs)




class DeprivationVariable(RwiVariable):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        output_filename = f"{self.component}.tif"
        self.local_path = os.path.join(os.path.dirname(geopackage_path), self.component, output_filename)



    def _compute_affected_(self):
        project = Project(os.getcwd())
        if geo.is_raster(self.local_path) and project.raster_mask is not None:


            affected_local_path = self.affected_path
            ds = Calc(calc='local_path*mask', outfile=affected_local_path, projectionCheck=True, format='GTiff',
                      creation_options=GTIFF_CREATION_OPTIONS, quiet=False, overwrite=True,
                      NoDataValue=None,
                      local_path=self.local_path, mask=project.raster_mask)
            ds = None
            assert os.path.exists(self.affected_path), f'Failed to compute {self.affected_path}'
            return affected_local_path

    def download(self, force=False, **kwargs):
        project = Project(os.getcwd())
        if os.path.exists(self.local_path) and not force:
            if project.raster_mask is not None:
                assert os.path.exists(self.affected_path), (f'The affected version of {self.component} force not exist.'
                                                            f'Consider assessing using --force flag')
            return self.local_path

        logger.info(f'Downloading {self.component}')
        out_folder, file_name = os.path.split(self.local_path)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        progress = kwargs.pop('progress', None)
        local_path = os.path.join(project.data_folder, self.component, f'{self.component}_downloaded.tif')
        download_raster(
            src_dataset_path=self.source,
            src_band=1,
            dst_dataset_path=local_path,
            polygons_dataset_path=project.geopackage_file_path,
            polygons_layer_name=project.polygons_layer_name,
            progress=progress

        )

        with gdal.config_options({
            'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES':'YES',
            'GDAL_CACHEMAX':'512',
            'GDAL_NUMTHREADS':'4'
        }):
            geo.import_raster(
                source=local_path, dst=self.local_path, target_srs=project.target_srs,
                crop_ds=project.geopackage_file_path, crop_layer_name=project.polygons_layer_name,
                return_handle=False,
                ## warpOptions come now
                warpMemoryLimit=1024,

            )
        if os.path.exists(local_path):
            os.remove(local_path)
        self._compute_affected_()


    def compute(self, force=True, **kwargs):
        assert force, f'invalid force={force}'
        return self.download(force=force, **kwargs)