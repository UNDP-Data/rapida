import os
from abc import ABC
from typing import List
from pathlib import Path
from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.project import Project
from cbsurge.session import Session
from cbsurge.util.http_get_json import http_get_json
import httpx
import logging


logger = logging.getLogger(__name__)

class RwiComponent(Component, ABC):

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
                v = RwiVariable(name=var_name,
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



class RwiVariable(Variable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def download(self, **kwargs):
        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        fgb_url = self.source
        logger.info(f'Downloading {fgb_url} to {geopackage_path}')
        pass

    def evaluate(self, **kwargs):
        pass

    def compute(self, **kwargs):
        pass

    def resolve(self, **kwargs):
        pass