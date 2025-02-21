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
import httpx
import logging



from cbsurge.core.component import Component



logger = logging.getLogger(__name__)

class ElectricityComponent(Component, ABC):

    dataset_url = 'https://geohub.data.undp.org/api/datasets/310aadaa61ea23811e6ecd75905aaf29'

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



class ElectricityVariable(Variable, ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        project = Project(path=os.getcwd())
        print(project.geopackage_file_name)

    def download(self, **kwargs):
        # grid_component = ElectricityComponent()
        # url = grid_component.get_url
        url = self.source
        # TODO: download data within polygons layer in geopackage
        self.download_geodata_by_admin(dataset_url=url)
        data = http_get_json(url)
        return data

    def evaluate(self, **kwargs):
        pass

    def compute(self, **kwargs):
        pass



ADMIN_DATASET = '/home/thuha/Desktop/UNDP/geo-cb-surge/data/kenya_adm2.geojson'
GRID_DATASET = '/home/thuha/Desktop/UNDP/geo-cb-surge/data/kenya_grid.fgb'
CLIP_ADMIN_PATH = '/home/thuha/Desktop/UNDP/geo-cb-surge/data/napak_moroto.fgb'
CLIP_GRID_PATH_V1 = '/home/thuha/Desktop/UNDP/geo-cb-surge/data/napak_moroto_grid_v1.fgb'
CLIP_GRID_PATH_V2 = '/home/thuha/Desktop/UNDP/geo-cb-surge/data/napak_moroto_grid_v2.fgb'


def clip_v1():
    admin_dataset = gpd.read_file(ADMIN_DATASET)
    grid_dataset = gpd.read_file(GRID_DATASET)
    napak_moroto_admin = admin_dataset[admin_dataset['name'].isin(['Napak', 'Moroto'])]
    napak_moroto_admin.to_file(CLIP_ADMIN_PATH, driver="FlatGeobuf")
    grid_dataset = grid_dataset.to_crs(napak_moroto_admin.crs)
    clipped_grid = grid_dataset.clip(mask=napak_moroto_admin.geometry, keep_geom_type=True, )
    clipped_grid.to_file(CLIP_GRID_PATH_V1, driver="FlatGeobuf")

def clip_v2():
    admin_dataset = gpd.read_file(ADMIN_DATASET)
    grid_dataset = gpd.read_file(GRID_DATASET)
    filtered_gdf = admin_dataset[admin_dataset['name'].isin(['Napak', 'Moroto'])]
    filtered_gdf.to_file(CLIP_ADMIN_PATH, driver="FlatGeobuf")
    grid_dataset = grid_dataset.to_crs(filtered_gdf.crs)
    clipped_grid = gpd.overlay(grid_dataset, filtered_gdf, how='intersection')
    clipped_grid.to_file(CLIP_GRID_PATH_V2, driver="FlatGeobuf")


def compute_grid_length(output_path, output_crs='ESRI:54009'):
    """
    Creates a FlatGeobuf file of admin units with total line length per unit.
    For each of the admin units, calculate the total length of the lines
    within the admin unit boundaries. Lines intersecting multiple units are split
    at the boundaries.

    Parameters:
    - output_path (str): Path for the output FlatGeobuf file.
    - output_crs (str): Projected CRS for accurate length calculation (default: EPSG:54009).
    """

    # Load datasets and reproject to output CRS
    grid_dataset_df = gpd.read_file(GRID_DATASET).to_crs(output_crs)
    admin_dataset_df = gpd.read_file(ADMIN_DATASET).to_crs(output_crs)

    # Spatial join: split lines at admin boundaries
    split_lines = gpd.overlay(grid_dataset_df, admin_dataset_df, how='intersection')

    # Calculate length for each split line
    split_lines['electricity_grid_length'] = np.divide(split_lines.geometry.length, 1000)  # in km

    # Aggregate total length per admin unit
    length_per_admin = split_lines.groupby('name', as_index=False)['electricity_grid_length'].sum()

    # Merge total lengths back with admin dataset
    admin_with_length = admin_dataset_df.merge(length_per_admin, on='name', how='left')
    admin_with_length['electricity_grid_length'] = admin_with_length['electricity_grid_length'].fillna(0)

    # Save result to FlatGeobuf
    admin_with_length.to_file(output_path, driver='FlatGeobuf')



if __name__ == "__main__":

    compute_grid_length(
        output_path='/home/thuha/Desktop/UNDP/geo-cb-surge/data/kenya_electricity_adm2.fgb'
    )
