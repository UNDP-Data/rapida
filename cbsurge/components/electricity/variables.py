from collections import OrderedDict
import logging


logger = logging.getLogger(__name__)

def generate_variables():
    """
    Generate electricity variables dict
    :return:
    """

    variables = OrderedDict()

    # for operator in ['sum', 'count', 'divide']:
    for operator in ['sum', 'count']:
        name = operator
        if operator == 'sum':
            name = 'length'
        # if operator == 'divide':
        #     name = 'density'
        variables[f'electricity_{name}'] = dict(
            title=f'Total {name} of electricity',
            source='geohub:/api/datasets/310aadaa61ea23811e6ecd75905aaf29',
            operator=operator,
            percentage=True,
        )
    #dependencies
    # variables['electricity_grid_length'] = dict(
    #     title='Total length of electricity grid',
    #     source='geohub:/api/datasets/310aadaa61ea23811e6ecd75905aaf29',
    #     operator='length',
    # )
    # variables['electricity_grid_density'] = dict(
    #     title='Density of electricity grid',
    #     source='geohub:/api/datasets/310aadaa61ea23811e6ecd75905aaf29',
    #     operator='density',
    # )
    # variables['electricity_grid_percentage'] = dict(
    #     title='Percentage of electricity grid',
    #     source='geohub:/api/datasets/310aadaa61ea23811e6ecd75905aaf29',
    #     operator='percentage',
    # )
    return variables