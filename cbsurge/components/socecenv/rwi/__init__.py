import os
import logging
from typing import List
from pathlib import Path
from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.project import Project
from cbsurge.session import Session
from cbsurge.util.download_geodata import download_geodata_by_admin
from cbsurge.util.resolve_url import resolve_geohub_url


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

            progress = kwargs.get('progress', None)
            if progress:
                variable_task = progress.add_task(
                    description=f'[red]Going to process {nvars} variables', total=nvars)

            for var_name in variables:
                var_data = variables_data[var_name]
                var_data['source'] = resolve_geohub_url(var_data['source'], 'flatgeobuf')

                # create instance
                v = RwiVariable(name=var_name,
                                       component=self.component_name,
                                       **var_data)
                # assess
                v(**kwargs)

                if variable_task and progress:
                    progress.update(variable_task, advance=1, description=f'Assessed {var_name}')




class RwiVariable(Variable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def download(self, **kwargs):
        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        fgb_url = self.source
        logger.info(f'Downloading {fgb_url} to {geopackage_path}')
        download_geodata_by_admin(
            dataset_url=fgb_url,
            geopackage_path=geopackage_path,
            progress=kwargs.get('progress', None)
        )

    def compute(self, **kwargs):
        pass

    def resolve(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass