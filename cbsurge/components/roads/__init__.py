import logging
from typing import List
from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.session import Session
from cbsurge.util.resolve_url import resolve_geohub_url


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

    def download(self, variables: List[str] = None, **kwargs) -> List[str]:
        pass


class RoadsVariable(Variable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def download(self, **kwargs):
        pass

    def compute(self, **kwargs):
        pass

    def resolve(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass