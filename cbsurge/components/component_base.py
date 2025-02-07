from typing import List, Dict
from cbsurge.session import Session
import logging
from abc import abstractmethod

logger = logging.getLogger(__name__)

class ComponentBase():
    """
    A base class for a component.
    Each component class should have the following methods to be implemented:

    - download: Download files for a component
    - assess: A command to do all processing for a component including downloading component data, merging and making stats for a given admin data.
    """

    def __init__(self, **kwargs):
        self.component_name = self.__class__.__name__.lower().split('component')[0]
        super().__init__(**kwargs)

    @property
    def variables(self) -> List[str]:
        with Session() as ses:
            return ses.get_variables(component=self.component_name)


    # @property
    # def variables(self) -> Dict[str, SurgeVariable]:
    #     with Session() as ses:
    #         vrs = ses.get_component(self.component_name)
    #         variables = dict()
    #         for var_name, var_data in vrs.items():
    #             v = SurgeVariable(name=var_name, component=self.component_name, **var_data)
    #             variables[var_name] = v
    #         return variables




    def download(self, variables: List[str] = None, **kwargs) -> List[str]:
        """
        Iterate over variables and download one by one
        :param variables:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def __call__(self, **kwargs):

        pass


