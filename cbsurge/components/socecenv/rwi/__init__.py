import os
import logging
from typing import List
from pathlib import Path
import geopandas as gpd
from rich.progress import Progress

from cbsurge import constants
from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.project import Project
from cbsurge.session import Session
from cbsurge.util.download_geodata import download_raster
from cbsurge.util.resolve_url import resolve_geohub_url
from cbsurge.stats.zst import zonal_stats


logger = logging.getLogger(__name__)


class RwiComponent(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # get parent package name for electricity
        current_dir = Path(__file__).resolve().parent
        parent_package_name = current_dir.parents[0].name

        self.component_name = f"{parent_package_name}.{ self.__class__.__name__.lower().split('component')[0]}"

    def __call__(self, variables: List[str] = None, **kwargs) -> str:

        logger.debug(f'Assessing component "{self.component_name}" ')
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if not var_name in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        project = Project(path=os.getcwd())

        with Session() as ses:
            variables_data = ses.get_component(self.component_name)
            nvars = len(variables)

            for var_name in variables:
                var_data = variables_data[var_name]
                var_data['source'] = resolve_geohub_url(var_data['source'])

                # create instance
                v = RwiVariable(name=var_name, component=self.component_name, **var_data)
                # assess
                v(**kwargs)


class RwiVariable(Variable):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def download(self, **kwargs):
        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        cog_url = self.source
        output_filename = f"{self.component}.tif"
        rwi_path = os.path.join(os.path.dirname(geopackage_path), output_filename)

        if os.path.exists(rwi_path):
            self.local_path = rwi_path
        else:
            self.local_path = download_raster(
                dataset_url=cog_url,
                geopackage_path=geopackage_path,
                output_filename=output_filename,
                progress=kwargs.get('progress', None)
            )

    def compute(self, **kwargs):
        pass

    def resolve(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
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

            gdf = zonal_stats(src_rasters=src_rasters,
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