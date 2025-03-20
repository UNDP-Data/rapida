import logging
import os
from typing import List

from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.project import Project
from cbsurge.session import Session

logger = logging.getLogger('rapida')

class BuiltupComponent(Component):
    def __call__(self, variables: List[str], **kwargs):
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if var_name not in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        project = Project(path=os.getcwd())
        logger.debug(f'Assessing component "{self.component_name}" in  {", ".join(project.countries)}')
        with Session() as session:
            variable_data = session.get_component(self.component_name)

            for var_name in variables:
                var_data = variable_data[var_name]

                v = BuiltupVariable(
                    name=var_name,
                    component=self.component_name,
                    **var_data
                )
                v()



class BuiltupVariable(Variable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        project = Project(path=os.getcwd())
        output_filename = f"{self.component}.tif"

    def __call__(self, *args, **kwargs):
        force_compute = kwargs.get('force_compute', False)
        progress = kwargs.get('progress', False)



    def download(self):
        pass

    def compute(self):
        pass

    def evaluate(self):
        pass

    def resolve(self):
        pass

