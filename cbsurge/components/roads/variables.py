from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def generate_variables():
    """
    Generate roads variables dict
    :return Roads variables definition
    """

    variables = OrderedDict()

    variables[f'roads_length'] = dict(
        title=f'Total length of roads',
        source='geohub:/api/datasets/300da70781b7a53808aab824543e6c2b',
        operator="sum",
        percentage=True,
    )
    return variables