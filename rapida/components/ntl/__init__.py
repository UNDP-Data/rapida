from rapida.core.component import Component
from rapida.core.variable import Variable
from rapida.project.project import Project
from rapida.session import Session
import logging
import os

logger = logging.getLogger('rapida')

class NtlComponent(Component):

    def __call__(self, variables: list[str],  **kwargs):
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if var_name not in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        with Session() as session:
            variable_data = session.get_component(self.component_name)

            for var_name in variables:
                var_data = variable_data[var_name]

                v = NTLVariable(
                    name=var_name,
                    component=self.component_name,

                    **var_data
                )
                v(**kwargs)


class NTLVariable(Variable):


    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        output_filename = f"{self.name}.tif"
        self.local_path = os.path.join(os.path.dirname(geopackage_path), self.component, output_filename)

    def download(self,force=False, **kwargs):
        pass

    def download(self, **kwargs):
        pass
    def resolve(self, **kwargs):
        pass

    def compute(self, **kwargs):
        pass
    def evaluate(self, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        print(self.name)