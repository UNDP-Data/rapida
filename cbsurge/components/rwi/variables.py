from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def generate_variables():
    """
    Generate relative wealth index variables dict
    :return RWI variables definition
    """

    variables = OrderedDict()

    for operator in ['mean', 'min', 'max']:
        variables[f'rwi_{operator}'] = dict(
            title=f'{operator} of relative wealth index',
            source='geohub:/api/datasets/fcde6ab53a79657a27906b3248a1979d',
            operator=operator
        )
    return variables