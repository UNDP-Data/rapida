import os
import logging
from typing import List
import geopandas as gpd
from rich.progress import Progress
from cbsurge import constants
from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.project import Project
from cbsurge.session import Session
from cbsurge.util import geo
from cbsurge.util.download_geodata import download_raster, download_rasta
from cbsurge.util.resolve_url import resolve_geohub_url
from cbsurge.stats.zst import zst
from osgeo_utils.gdal_calc import Calc
from cbsurge.constants import GTIFF_CREATION_OPTIONS
from osgeo import gdal
logger = logging.getLogger(__name__)


class RwiComponent(Component):

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
                var_data['source'] = resolve_geohub_url(var_data['source'])

                # create instance
                v = RwiVariable(name=var_name, component=self.component_name, **var_data)
                # assess
                v(**kwargs)


class RwiVariable(Variable):

    @property
    def affected_path(self):
        path, file_name = os.path.split(self.local_path)
        fname, ext = os.path.splitext(file_name)
        return os.path.join(path, f'{fname}_affected{ext}')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        output_filename = f"{self.component}.tif"
        self.local_path = os.path.join(os.path.dirname(geopackage_path), self.component, output_filename)
        #self.local_path = os.path.join(os.path.dirname(geopackage_path), output_filename)

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



    def _compute_affected_(self):
        if geo.is_raster(self.local_path):

            project = Project(os.getcwd())
            affected_local_path = self.affected_path
            ds = Calc(calc='local_path*mask', outfile=affected_local_path, projectionCheck=True, format='GTiff',
                      creation_options=GTIFF_CREATION_OPTIONS, quiet=False, overwrite=True,
                      NoDataValue=None,
                      local_path=self.local_path, mask=project.raster_mask)
            ds = None
            assert os.path.exists(self.affected_path), f'Failed to compute {self.affected_path}'
            return affected_local_path

    def download(self, force_compute=False, **kwargs):

        if os.path.exists(self.local_path) and not force_compute:

            assert os.path.exists(self.affected_path), (f'The affected version of {self.component} force not exist.'
                                                        f'Consider assessing using --force_compute flag')
            return self.local_path

        logger.info(f'Downloading {self.component}')
        out_folder, file_name = os.path.split(self.local_path)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        project = Project(os.getcwd())
        progress = kwargs.pop('progress', None)
        local_path = os.path.join(project.data_folder, self.component, f'{self.component}_downloaded.tif')
        download_rasta(
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


    def compute(self, force_compute=True, **kwargs):
        assert force_compute, f'invalid force_compute={force_compute}'
        return self.download(force_compute=force_compute, **kwargs)

    # def download(self, **kwargs):
    #     """
    #     Download RWI data source
    #     """
    #     project = Project(path=os.getcwd())
    #     geopackage_path = project.geopackage_file_path
    #     cog_url = self.source
    #
    #     force_compute = kwargs.get('force_compute', False)
    #
    #     if force_compute == True or not os.path.exists(self.local_path):
    #         self.local_path = download_raster(
    #             dataset_url=cog_url,
    #             geopackage_path=geopackage_path,
    #             output_filename=os.path.basename(self.local_path),
    #             progress=kwargs.get('progress', None)
    #         )
    #
    #     if project.raster_mask is not None and geo.is_raster(self.local_path):
    #         affected_local_path = self.affected_path
    #         if force_compute == True or not os.path.exists(affected_local_path):
    #             geo.clip_raster_with_mask(
    #                 source=self.local_path,
    #                 mask=project.raster_mask,
    #                 output_path=affected_local_path,
    #                 progress=kwargs.get('progress', None)
    #             )

    # def compute(self, **kwargs):
    #     pass

    def resolve(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        """
        make zonal statistics for variable. compute affected variable if project has mask layer.
        """
        progress: Progress = kwargs.get('progress', Progress())
        evaluate_task = None
        if progress is not None:
            evaluate_task = progress.add_task(
                description=f'[red]Going to evaluate {self.name} in {self.component} component', total=None)

        dst_layer = f'stats.{self.component}'
        project = Project(path=os.getcwd())
        layers = gpd.list_layers(project.geopackage_file_path)
        lnames = layers.name.tolist()
        if dst_layer in lnames:
            polygons_layer = dst_layer
        else:
            polygons_layer = constants.POLYGONS_LAYER_NAME

        if self.operator:
            assert os.path.exists(self.local_path), f'{self.local_path} does not exist'

            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task, description=f'Evaluating variable {self.name} using zonal stats')

            # raster variable, run zonal stats
            src_rasters = [self.local_path]
            var_ops = [(self.name, self.operator)]

            if project.raster_mask is not None:
                affected_local_path = self.affected_path
                src_rasters.append(affected_local_path)
                var_ops.append((f'{self.name}_affected', self.operator))

            gdf = zst(src_rasters=src_rasters,
                      polygon_ds=project.geopackage_file_path,
                      polygon_layer=polygons_layer, vars_ops=var_ops
                      )

            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task, description=f'Evaluated variable {self.name} using zonal stats')

            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task,
                                description=f'Writing {self.name} to {project.geopackage_file_path}:{dst_layer}')

            gdf.to_file(project.geopackage_file_path, layer=dst_layer, driver="GPKG")
        else:
            progress.update(evaluate_task,
                            description=f'{self.name} was skipped because of lack of operator definition.')

        if progress and evaluate_task:
            progress.remove_task(evaluate_task)
