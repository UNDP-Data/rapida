import os
from abc import ABC
from typing import List
from pathlib import Path

import geopandas as gpd
import numpy as np

from cbsurge.core.variable import Variable
from cbsurge.project import Project
from cbsurge.session import Session
from cbsurge.util.http_get_json import http_get_json
from cbsurge.util.download_geodata import download_geodata_by_admin
import httpx
import logging

from cbsurge.core.component import Component


logger = logging.getLogger(__name__)

class ElectricityComponent(Component, ABC):

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

        progress = kwargs.get('progress', None)
        project = Project(path=os.getcwd())

        with Session() as ses:
            variables_data = ses.get_component(self.component_name)
            nvars = len(variables)

            if progress:
                variable_task = progress.add_task(
                    description=f'[red]Going to process {nvars} variables', total=nvars)

            for var_name in variables:
                var_data = variables_data[var_name]

                if var_data['source'] and var_data['source'].startswith('geohub:'):
                    geohub_endpoint = ses.get_config_value_by_key('geohub_endpoint')
                    var_data['source'] = var_data['source'].replace('geohub:', geohub_endpoint)
                    var_data['source'] = self.get_url(var_data['source'])

                # create instance
                v = ElectricityVariable(name=var_name,
                                       component=self.component_name,
                                       **var_data)
                # assess
                v(**kwargs)

                if variable_task and progress:
                    progress.update(variable_task, advance=1, description=f'Assessed {var_name}')

    def get_url(self, dataset_url):
        try:
            timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
            data = http_get_json(url=dataset_url, timeout=timeout)
            for link in data['properties']['links']:
                if link['rel'] == 'flatgeobuf':
                    return link['href']
        except Exception as e:
            logger.error(f'Failed to get electricity grid from  {dataset_url}. {e}')
            raise



class ElectricityVariable(Variable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def affected_layer_name(self):
        return f'{self.name}_affected'

    def download(self, **kwargs):
        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        download_geodata_by_admin(
            dataset_url=self.source,
            geopackage_path=geopackage_path,
            layer_name=self.component
        )

    def evaluate(self, **kwargs):


        logger.debug(f'Evaluating variable variable "{self.name}"')
        destination_layer = f'stats.{self.component}'
        project = Project(path=os.getcwd())

        if destination_layer in gpd.list_layers(project.geopackage_file_path):
            polygons_layer = destination_layer
        else:
            polygons_layer = 'polygons'
        polygons_df = gpd.read_file(project.geopackage_file_path, layer=polygons_layer)

        if self.name in polygons_df.columns: # remove the variable column if it exists
            polygons_df.drop(columns=[self.name], inplace=True)

        grid_layer_df = gpd.read_file(project.geopackage_file_path, layer=self.component)

        electricity_grid_df = gpd.overlay(grid_layer_df, polygons_df, how='intersection')

        # write split lines to local path
        self.local_path = os.path.join(project.data_folder, self.name)
        electricity_grid_df.to_file(f'{self.local_path}.fgb', driver='FlatGeobuf', layer='masked_grid')

        electricity_grid_df[self.name] = electricity_grid_df.geometry.length
        length_per_unit = electricity_grid_df.groupby('name', as_index=False)[self.name].sum()

        admin_with_length = polygons_df.merge(length_per_unit, on='name', how='left')
        admin_with_length[self.name] = admin_with_length[self.name].fillna(0)
        admin_with_length.to_file(project.geopackage_file_path, driver='GPKG', layer=destination_layer, mode='w')



        if project.vector_mask is not None:  # if the mask is provided then compute the affected
            mask_df = gpd.read_file(project.geopackage_file_path, layer=project.vector_mask)
            grid_layer_df_affected = gpd.overlay(grid_layer_df, mask_df, how='intersection')
            grid_layer_df_affected.to_file(project.geopackage_file_path, driver='GPKG', layer=self.affected_layer_name, mode='w')
            grid_layer_df_affected[f'{self.name}_affected'] = grid_layer_df_affected.geometry.length

            # drop the affected column if it exists
            if self.affected_layer_name in polygons_df.columns:
                polygons_df.drop(columns=[self.affected_layer_name], inplace=True)
            # overlay the affected with the admin
            masked_electricity_grid_df = gpd.overlay(grid_layer_df_affected, polygons_df, how='intersection')
            masked_electricity_grid_df[self.affected_layer_name] = masked_electricity_grid_df.geometry.length
            affected_length_per_unit = masked_electricity_grid_df.groupby('name', as_index=False)[f'{self.name}_affected'].sum()
            admin_with_length = admin_with_length.merge(affected_length_per_unit, on='name', how='left')
            admin_with_length[f'{self.name}_affected'] = admin_with_length[f'{self.name}_affected'].fillna(0)

            # compute percentage of affected
            admin_with_length[f'{self.name}_affected_percent'] = np.divide(admin_with_length[f'{self.name}_affected'], admin_with_length[self.name])

            # compute area of admin
            admin_with_length['area'] = admin_with_length.geometry.area

            # density: affected per unit area of admin unit
            admin_with_length[f'{self.name}_affected_density'] = np.divide(admin_with_length[f'{self.name}_affected'], admin_with_length['area'])

            # drop the area column
            admin_with_length.drop(columns=['area'], inplace=True)

            admin_with_length.to_file(project.geopackage_file_path, driver='GPKG', layer=destination_layer, mode='w')





    def _compute_affected(self):
        pass

    def compute(self, **kwargs):
        pass

    def resolve(self, **kwargs):
        pass



if __name__ == "__main__":
    va = ElectricityVariable(
        name='electricity_grid_length',
        component='builtenv.electricity',
        source='geohub:https://geohub.data.undp.org/api/datasets/310aadaa61ea23811e6ecd75905aaf29',
        title='Electricity Grid Length',
    )
    va.compute()