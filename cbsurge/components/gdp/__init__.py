import os
import re
import logging
import math
import shutil
from typing import List
from rich.progress import Progress
import geopandas as gpd
from osgeo import gdal, ogr
from osgeo_utils.gdal_calc import Calc
from cbsurge import constants
from cbsurge.constants import GTIFF_CREATION_OPTIONS
from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.project.project import Project
from cbsurge.session import Session
from cbsurge.stats.raster_zonal_stats import zst
from cbsurge.util import geo
from cbsurge.util.download_geodata import download_raster
logger = logging.getLogger(__name__)


class GdpComponent(Component):
    """
    Gridded GDP component

    License: Creative Commons Attribution 4.0 International
    Citation: Wang, T., & Sun, F. (2023). Global gridded GDP under the historical and future scenarios [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7898409
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def __call__(self, variables: List[str] = None, target_year: int=None, **kwargs) -> str:
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if not var_name in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        with Session() as ses:
            variables_data = ses.get_component(self.component_name)

            # delete stats layer if exist
            project = Project(path=os.getcwd())
            geopackage_path = project.geopackage_file_path
            with ogr.Open(geopackage_path, 1) as ds:
                dst_layer = f"stats.{self.component_name}"
                layer_index = ds.GetLayerByName(dst_layer)
                if layer_index is not None:
                    ds.DeleteLayer(dst_layer)

            for var_name in variables:
                var_data = variables_data[var_name]
                sources = self.interpolate_data_url(var_data['source'], target_year=target_year)

                # create instance
                v = GdpVariable(name=var_name,
                                component=self.component_name,
                                sources=sources,
                                target_year=target_year,
                                **var_data)
                # assess
                v(**kwargs)

    def interpolate_data_url(self, source: str, target_year: int):
        """
        Interpolate data from source to target year.
        """
        with Session() as session:
            hostname = session.get_blob_service_account_url()
            parts = source.split(":", 2)
            pathname = parts[2]
            source = f"{hostname}/{pathname}"

        if 2000 <= target_year <= 2020:
            return [source.replace("{year}", str(target_year))]

        if target_year < 2000:
            return [source.replace("{year}", "2000")]

        if target_year > 2100:
            return [source.replace("{year}", "2100/ssp2")]

        lower_year = math.floor(target_year / 5) * 5
        upper_year = lower_year + 5

        if target_year % 5 == 0:
            return [source.replace("{year}", f"{target_year}/ssp2")]

        return [
            source.replace("{year}", f"{lower_year}/ssp2"),
            source.replace("{year}", f"{upper_year}/ssp2"),
        ]




class GdpVariable(Variable):

    @property
    def variable_name(self) -> str:
        return f"{self.name}_{self.target_year}"

    @property
    def affected_variable(self) -> str:
        return f"{self.variable_name}_affected"

    @property
    def affected_percentage_variable(self) -> str:
        return f"{self.affected_variable}_percentage"

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
        self.local_path = os.path.join(os.path.dirname(geopackage_path), self.component,str(self.target_year), output_filename)

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
        progress: Progress = kwargs.get('progress', Progress())
        variable_task = None
        if progress is not None:
            variable_task = progress.add_task(
                description=f'[blue] Assessing {self.component}->{self.name}', total=None)

        self.download(**kwargs)
        if progress is not None and variable_task is not None:
            progress.update(variable_task, description=f'[blue] Downloaded {self.component}->{self.name}')

        self.evaluate(**kwargs)
        if progress is not None and variable_task is not None:
            progress.update(variable_task, description=f'[blue] Assessed {self.component}->{self.name}')


    def download(self, force_compute=False, **kwargs):
        project = Project(os.getcwd())
        progress: Progress = kwargs.get('progress', Progress())

        download_task = None
        if progress is not None:
            download_task = progress.add_task(
                description=f'[red] Downloading {self.name}', total=None)

        if force_compute == False and os.path.exists(self.local_path):
            if not os.path.exists(self.affected_path):
                self._compute_affected_()
            if progress is not None and download_task is not None:
                progress.update(download_task, description=f'[red] File: {self.local_path} exists. Downloading was skipped')
                progress.remove_task(download_task)
            return self.local_path

        local_sources = list()

        if progress is not None and download_task is not None:
            progress.update(download_task, description=f'[red] Start downloading sources', total=len(self.sources))
        for source in self.sources:
            src_path = source
            hostname, pathname = src_path.split('/gdp/', 1)
            local_path = os.path.join(project.data_folder, self.component, pathname)
            tmp_local_path = f"{local_path}.tmp"
            source_folder = os.path.dirname(tmp_local_path)
            if not os.path.exists(source_folder):
                os.makedirs(source_folder)
            logger.debug(f'Going to download {src_path} to {tmp_local_path}')

            download_raster(
                src_dataset_path=src_path,
                src_band=1,
                dst_dataset_path=tmp_local_path,
                polygons_dataset_path=project.geopackage_file_path,
                polygons_layer_name=project.polygons_layer_name,
                progress=progress
            )

            with gdal.config_options({
                'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES',
                'GDAL_CACHEMAX': '512',
                'GDAL_NUMTHREADS': '4'
            }):
                geo.import_raster(
                    source=tmp_local_path, dst=local_path, target_srs=project.target_srs,
                    crop_ds=project.geopackage_file_path, crop_layer_name=project.polygons_layer_name,
                    return_handle=False,
                    ## warpOptions come now
                    warpMemoryLimit=1024,
                )
            if os.path.exists(tmp_local_path):
                os.remove(tmp_local_path)

            local_sources.append(local_path)

            if progress is not None and download_task is not None:
                progress.update(download_task, description=f'[red] Downloaded {local_path}', advance=1)

        target_file = local_sources[0]
        if len(local_sources) == 2:
            if progress is not None and download_task is not None:
                progress.update(download_task, description=f'[red] Interpolating data for {self.target_year}', total=None)
            # interpolate target year data from two
            target_file = self.interpolate_target_year(local_sources)

            if progress is not None and download_task is not None:
                progress.update(download_task, description=f'[red] Interpolated data for {self.target_year}')

        shutil.move(target_file, self.local_path)
        shutil.rmtree(os.path.dirname(target_file))

        if progress and download_task:
            progress.remove_task(download_task)

        self._compute_affected_(**kwargs)

    def _compute_affected_(self, **kwargs):
        if geo.is_raster(self.local_path):
            project = Project(os.getcwd())
            if project.raster_mask:
                progress: Progress = kwargs.get('progress', Progress())

                affect_task = None
                if progress is not None:
                    affect_task = progress.add_task(
                        description=f'[red] masking downloaded data by affected area', total=None)


                affected_local_path = self.affected_path
                ds = Calc(calc='local_path*mask', outfile=affected_local_path, projectionCheck=True, format='GTiff',
                          creation_options=GTIFF_CREATION_OPTIONS, quiet=False, overwrite=True,
                          NoDataValue=None,
                          local_path=self.local_path, mask=project.raster_mask)
                ds = None
                assert os.path.exists(self.affected_path), f'Failed to compute {self.affected_path}'

                if progress is not None and affect_task is not None:
                    progress.update(affect_task, description=f'[red] computed masked data for affected area')
                    progress.remove_task(affect_task)
                return affected_local_path

    def interpolate_target_year(self, local_sources: List[str]):
        target_year = self.target_year

        year_pattern = re.compile(r"/gdp/(\d{4})/ssp2")
        years_paths = [(int(year_pattern.search(path).group(1)), path) for path in local_sources]
        (A_year, A_path), (B_year, B_path) = sorted(years_paths)

        output_file = A_path.replace(str(A_year), str(target_year))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # predict target year data by using linear regression from two years data
        ds = Calc(
            A=A_path,
            B=B_path,
            outfile=output_file,
            calc=f"A + (B - A) * (({target_year} - {A_year}) / ({B_year} - {A_year}))",
            projectionCheck=True,
            quiet=False,
            overwrite=True,
            format="GTiff",
            creation_options=GTIFF_CREATION_OPTIONS,
            NoDataValue=None
        )
        ds = None

        # cleanup
        for path in local_sources:
            year_folder = os.path.dirname(os.path.dirname(path))
            if os.path.exists(year_folder):
                shutil.rmtree(year_folder)

        return output_file

    def compute(self, force_compute=True, **kwargs):
        assert force_compute, f'invalid force_compute={force_compute}'
        return self.download(force_compute=force_compute, **kwargs)

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
                description=f'[red] Going to evaluate {self.variable_name} in {self.component} component', total=None)

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
                progress.update(evaluate_task, description=f'[red] Evaluating variable {self.variable_name} using zonal stats')

            # raster variable, run zonal stats
            src_rasters = [self.local_path]
            var_ops = [(self.variable_name, self.operator)]

            if project.raster_mask is not None:
                affected_local_path = self.affected_path
                src_rasters.append(affected_local_path)
                var_ops.append((self.affected_variable, self.operator))

            gdf = zst(src_rasters=src_rasters,
                      polygon_ds=project.geopackage_file_path,
                      polygon_layer=polygons_layer, vars_ops=var_ops
                      )

            if project.raster_mask:
                percentage = self.percentage
                if percentage:
                    gdf.eval(f"{self.affected_percentage_variable}={self.affected_variable}/{self.variable_name}*100",
                                   inplace=True)

            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task,
                                description=f'[red] Writing {self.variable_name} to {project.geopackage_file_path}:{dst_layer}')

            gdf.to_file(project.geopackage_file_path, layer=dst_layer, driver="GPKG", overwrite=True)
        else:
            progress.update(evaluate_task,
                            description=f'[red] {self.variable_name} was skipped because of lack of operator definition.')

        if progress and evaluate_task:
            progress.remove_task(evaluate_task)
