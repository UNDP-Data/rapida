import os
import logging
from typing import List
import geopandas as gpd
from rich.progress import Progress
from cbsurge import constants
from cbsurge.core.component import Component
from cbsurge.components.rwi import RwiVariable
from cbsurge.project import Project
from cbsurge.session import Session
from cbsurge.util import geo
from cbsurge.util.download_geodata import download_raster
from cbsurge.util.resolve_url import resolve_geohub_url
from cbsurge.stats.zst import zst


logger = logging.getLogger(__name__)


class DeprivComponent(Component):

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
                v = DeprivationVariable(name=var_name, component=self.component_name, **var_data)
                # assess
                v(**kwargs)


class DeprivationVariable(RwiVariable):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        output_filename = f"{self.component}.tif"
        self.local_path = os.path.join(os.path.dirname(geopackage_path), self.component, output_filename)