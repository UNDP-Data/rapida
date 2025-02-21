import json
from collections import OrderedDict
import logging

from cbsurge.util.setup_logger import setup_logger

logger = logging.getLogger(__name__)

def generate_variables():
    """
    Generate electricity variables dict
    :return:
    """

    variables = OrderedDict()

    #dependencies
    variables['electricity_grid'] = dict(
        title='Total length of electricity grid',
        source='geohub:/api/datasets/310aadaa61ea23811e6ecd75905aaf29',
        operator="sum"
    )
    return variables


if __name__ == '__main__':
    logger = setup_logger(name='rapida', level=logging.INFO)

    variables = generate_variables()
    print(json.dumps(variables, indent=2))