import logging
import os
from abc import ABC
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from osgeo import gdal, osr

from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.project import Project
from cbsurge.session import Session
from cbsurge.util.download_geodata import download_vector
from cbsurge.util.proj_are_equal import proj_are_equal
from cbsurge.util.resolve_url import resolve_geohub_url

logger = logging.getLogger(__name__)


def compute_grid_length(grid_df, polygons_df):
    project = Project(path=os.getcwd())
    # overlay with the polygons to get only the grid within the polygons
    electricity_grid_df = gpd.overlay(grid_df, polygons_df, how='intersection')
    electricity_grid_df['electricity_grid_length'] = electricity_grid_df.geometry.length
    length_per_unit = electricity_grid_df.groupby('h3id', as_index=False)['electricity_grid_length'].sum()
    output_df = polygons_df.merge(length_per_unit, on='h3id', how='left')
    output_df['electricity_grid_length'] = output_df['electricity_grid_length'].fillna(0)

    # affected_grid_length
    if project.vector_mask is not None:
        mask_df = gpd.read_file(project.geopackage_file_path, layer=project.vector_mask)
        # overlay with the mask to get only the affected grid
        electricity_grid_df_affected = gpd.overlay(electricity_grid_df, mask_df, how='intersection')
        electricity_grid_df_affected.to_file(project.geopackage_file_path, driver="GPKG", layer="affected_electricity")
        electricity_grid_df_affected['affected_electricity_grid_length'] = electricity_grid_df_affected.geometry.length
        # overlay the affected with the admin
        affected_length_per_unit = electricity_grid_df_affected.groupby('h3id', as_index=False)[
            'affected_electricity_grid_length'].sum()
        output_df = output_df.merge(affected_length_per_unit, on='h3id', how='left')
    return output_df


class ElectricityComponent(Component, ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # get parent package name for electricity
        current_dir = Path(__file__).resolve().parent
        parent_package_name = current_dir.parents[0].name

        self.component_name = f"{ self.__class__.__name__.lower().split('component')[0]}"

    def __call__(self, variables: List[str] = None, **kwargs) -> str:

        logger.debug(f'Assessing component "{self.component_name}" ')
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if not var_name in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return
        progress = kwargs.get('progress', None)
        with Session() as ses:
            variables_data = ses.get_component(self.component_name)
            nvars = len(variables)
            if progress:
                variable_task = progress.add_task(
                    description=f'[red]Going to process {nvars} variables', total=nvars)
            for var_name in variables:
                var_data = variables_data[var_name]

                var_data['source'] = resolve_geohub_url(dataset_url=var_data['source'], link_name='flatgeobuf')
                # create instance
                v = ElectricityVariable(name=var_name,
                                       component=self.component_name,
                                       **var_data)
                # assess
                v(**kwargs)

                if variable_task and progress:
                    progress.update(variable_task, advance=1, description=f'Assessed {var_name}')


class ElectricityVariable(Variable):

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
                    self.download(**kwargs)
                    if progress is not None and variable_task is not None:
                        progress.update(variable_task, description=f'[blue]Downloaded {self.component}->{self.name}')
                else:
                    self.compute(**kwargs)
                    if progress is not None and variable_task is not None:
                        progress.update(variable_task, description=f'[blue]Computed {self.component}->{self.name}')
            else:
                sources = self.resolve(**kwargs)

        self.evaluate(**kwargs)
        if progress is not None and variable_task is not None:
            progress._tasks[variable_task].total = 1
            progress.update(variable_task, description=f'[blue]Assessed {self.component}->{self.name}', advance=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def affected_name(self):
        return f'{self.name}_affected'

    @property
    def percentage(self):
        return self.__dict__.get('percentage', False)

    def download(self, **kwargs):
        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        force_compute = kwargs.get('force_compute', False)
        progress = kwargs.get('progress', False)
        layers = pyogrio.list_layers(geopackage_path)
        layer_names = layers[:, 0]
        if force_compute or self.component not in layer_names:
            with gdal.OpenEx(geopackage_path, gdal.OF_READONLY | gdal.OF_VECTOR) as poly_ds:
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
                    dst_dataset_path=geopackage_path,
                    dst_layer_name=self.component,
                    dst_srs=target_srs,
                    dst_layer_mode="w",
                    mask_polygons=mask_polygons,
                    progress=progress,
                    add_polyid=True
                )
                lyr.SetAttributeFilter(None)
                lyr.ResetReading()

    def grid_density(self, grid_df, polygons_df):
        project = Project(path=os.getcwd())
        length_gdf = compute_grid_length(grid_df=grid_df, polygons_df=polygons_df)
        length_gdf['area'] = polygons_df.geometry.area
        length_gdf[self.name] = np.divide(length_gdf['electricity_grid_length'], length_gdf['area'])

        # affected_grid_density
        if project.vector_mask is not None:
            length_gdf[f'affected_{self.name}'] = np.divide(length_gdf['affected_electricity_grid_length'], length_gdf['area'])
        output_gdf = length_gdf.drop(columns=['area'])
        return output_gdf

    def evaluate(self, **kwargs):
        logger.info(f'Evaluating variable "{self.name}"')
        destination_layer = f'stats.{self.component}'
        project = Project(path=os.getcwd())

        output_df = None

        if destination_layer in gpd.list_layers(project.geopackage_file_path):
            polygons_layer = destination_layer
        else:
            polygons_layer = 'polygons'
        polygons_df = gpd.read_file(project.geopackage_file_path, layer=polygons_layer)
        grid_df = gpd.read_file(project.geopackage_file_path, layer=self.component)
        if self.name in polygons_df.columns: # remove the variable column if it exists
            polygons_df.drop(columns=[self.name], inplace=True)

        if self.operator == 'density':
            output_df = self.grid_density(grid_df, polygons_df)

        if self.operator == 'length':
            output_df = compute_grid_length(grid_df, polygons_df)
        output_df.to_file(project.geopackage_file_path, driver='GPKG', layer=destination_layer, mode='w')




    def _compute_affected(self):
        pass

    def compute(self, force_compute=True, **kwargs):
        assert force_compute, f'invalid force_compute={force_compute}'
        return self.download(force_compute=force_compute, **kwargs)

    def resolve(self, **kwargs):
        pass