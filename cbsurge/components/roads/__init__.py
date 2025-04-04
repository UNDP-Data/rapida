import os
import logging
from typing import List
from rich.progress import Progress
from cbsurge import constants
from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.session import Session
from cbsurge.util.resolve_url import resolve_geohub_url
from cbsurge.util.download_geodata import download_vector
from cbsurge.util.proj_are_equal import proj_are_equal
from cbsurge.project import Project
from cbsurge.stats.vector_zonal_stats import vector_line_zonal_stats
from osgeo import gdal, osr
import pyogrio
import shapely
import geopandas as gpd


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

    @property
    def affected_layer(self):
       return f"{self.component}_affected"

    @property
    def affected_variable(self):
        return f"{self.name}_affected"

    @property
    def affected_percentage_variable(self):
        return f"{self.name}_affected_percentage"

    @property
    def percentage(self):
        return self.__dict__.get('percentage', False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        project = Project(path=os.getcwd())
        self.local_path = project.geopackage_file_path

    def download(self, **kwargs):
        project = Project(path=os.getcwd())
        self.local_path = project.geopackage_file_path

        force_compute = kwargs.get('force_compute', False)
        progress = kwargs.get('progress', False)

        layers = pyogrio.list_layers(self.local_path)
        layer_names = layers[:, 0]

        if force_compute == True or not self.component in layer_names:
            with gdal.OpenEx(project.geopackage_file_path, gdal.OF_READONLY | gdal.OF_VECTOR) as poly_ds:
                lyr = poly_ds.GetLayerByName(project.polygons_layer_name)

                try:
                    src_dataset_info = pyogrio.read_info(self.source)
                except Exception as e:
                    if '404' in str(e):
                        logger.error(f'{self.source} is not reachable')
                    raise
                src_crs = src_dataset_info['crs']
                src_srs = osr.SpatialReference()
                src_srs.SetFromUserInput(src_crs)
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

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
                    src_dataset_url=self.source,
                    src_layer_name=0,
                    dst_dataset_path=self.local_path,
                    dst_layer_name=self.component,
                    dst_srs=target_srs,
                    dst_layer_mode="w",
                    mask_polygons=mask_polygons,
                    progress=progress,
                    add_polyid=False
                )
                lyr.SetAttributeFilter(None)
                lyr.ResetReading()

        df_polygon = gpd.read_file(self.local_path, layer=project.polygons_layer_name)
        road_lines = gpd.read_file(self.local_path, layer=self.component)
        road_lines.sindex
        df_polygon.sindex
        poly_cols = df_polygon.columns.tolist()
        cols_to_drop = set(poly_cols).difference(['h3id']).difference(road_lines.columns.tolist())
        road_lines = gpd.overlay(road_lines, df_polygon, how="intersection", make_valid=True, keep_geom_type=True)
        road_lines.drop(columns=list(cols_to_drop), inplace=True)
        road_lines.rename(columns={'h3id': 'polyid'}, inplace=True)
        road_lines.to_file(self.local_path, driver='GPKG', layer=self.component, mode='w')

        if project.vector_mask is not None:
            if force_compute == True or self.affected_layer not in layer_names:
                df_line = gpd.read_file(self.local_path, layer=self.component)
                df_mask = gpd.read_file(self.local_path, layer=project.vector_mask)
                df_line_mask = gpd.clip(df_line, mask=df_mask, keep_geom_type=True)
                df_line_mask.to_file(self.local_path, driver='GPKG', layer=self.affected_layer, mode='w')

        return self.local_path

    def compute(self, force_compute=True, **kwargs):
        assert force_compute, f'invalid force_compute={force_compute}'
        return self.download(force_compute=force_compute, **kwargs)

    def resolve(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        progress: Progress = kwargs.get('progress', False)
        evaluate_task = None
        if progress is not None:
            evaluate_task = progress.add_task(
                description=f'[green]Going to evaluate {self.name} in {self.component} component', total=None)

        destination_layer = f'stats.{self.component}'
        project = Project(path=os.getcwd())

        layers = pyogrio.list_layers(self.local_path)
        layer_names = layers[:, 0]

        if progress is not None and evaluate_task is not None:
            progress.update(evaluate_task, description=f'[green]Loading layers...')

        if destination_layer in layer_names:
            polygons_layer = destination_layer
        else:
            polygons_layer = constants.POLYGONS_LAYER_NAME

        df_polygon = gpd.read_file(self.local_path, layer=polygons_layer)
        df_line = gpd.read_file(self.local_path, layer=self.component)

        for col in [self.name, self.affected_variable, self.affected_percentage_variable]:
            if col in df_polygon.columns:
                df_polygon.drop(columns=[col], inplace=True)

        assert self.operator is not None, f"[green]Operator is missing for variable {self.name}"

        if self.operator:
            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task, description=f'[green]Computing {self.name}.')

            output_df = vector_line_zonal_stats(
                df_polygon=df_polygon,
                df_line=df_line,
                operator=self.operator,
                field_name=self.name
            )

            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task, description=f'[green]Computed {self.name}.')

            if project.vector_mask is not None and self.affected_layer in layer_names:
                if progress is not None and evaluate_task is not None:
                    progress.update(evaluate_task, description=f'[green]Computing {self.affected_layer}.')

                df_line_affected = gpd.read_file(self.local_path, layer=self.affected_layer)
                output_df = vector_line_zonal_stats(
                    df_polygon=output_df,
                    df_line=df_line_affected,
                    operator=self.operator,
                    field_name=self.affected_variable
                )

                if progress is not None and evaluate_task is not None:
                    progress.update(evaluate_task, description=f'[green]Computed {self.affected_layer}.')

                percentage = self.percentage
                if percentage:
                    output_df.eval(f"{self.affected_percentage_variable}={self.affected_variable}/{self.name}*100", inplace=True)
                    if progress is not None and evaluate_task is not None:
                        progress.update(evaluate_task, description=f'[green]Computed {self.affected_variable}.')

            output_df.to_file(self.local_path, driver='GPKG', layer=destination_layer, mode='w')

            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task, description=f'[green]updated variables to {self.local_path} as {destination_layer}')
        else:
            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task, description=f"[red]variable {self.name} has no operator is specified. It was skipped.")

        if progress is not None and evaluate_task:
            progress.remove_task(evaluate_task)
