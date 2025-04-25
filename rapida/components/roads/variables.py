from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def generate_variables():
    """
    Generate roads variables dict
    :return Roads variables definition
    """

    variables = OrderedDict()

    license = "Creative Commons Zero 1.0 Universal"
    attribution = "Global biodiversity model for policy support, GLOBIO"

    for operator in ['sum', 'count']:
        name = operator
        if operator == 'sum':
            name = "length"
        variables[f'roads_{name}'] = dict(
            title=f'Total {name} of roads',
            source='geohub:/api/datasets/300da70781b7a53808aab824543e6c2b',
            operator=operator,
            percentage=True,
            license=license,
            attribution=attribution,
        )

    return variables