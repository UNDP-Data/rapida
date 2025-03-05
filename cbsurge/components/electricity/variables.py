from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def generate_variables():
    """
    Generate electricity variables dict
    :return:
    """

    variables = OrderedDict()

    #dependencies
    variables['electricity_grid_length'] = dict(
        title='Total length of electricity grid',
        source='geohub:/api/datasets/310aadaa61ea23811e6ecd75905aaf29',
        operator='length',
        percentage=True
    )
    variables['electricity_grid_density'] = dict(
        title='Density of electricity grid',
        source='geohub:/api/datasets/310aadaa61ea23811e6ecd75905aaf29',
        operator='density',
    )
    return variables