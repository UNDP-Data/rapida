from abc import abstractmethod
from typing import List

from cbsurge.session import Session


class Component:
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


    @abstractmethod
    def __call__(self, **kwargs):

        pass

