from cbsurge.components.component_base import ComponentBase
from cbsurge.session import Session

class PopulationComponent(ComponentBase):

    def __init__(self, year=2020, country=None):
        with Session() as ses:
            print()