from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def generate_variables():
    """
    Generate relative wealth index variables dict
    :return:
    """

    variables = OrderedDict()

    #dependencies
    variables['rwi_mean'] = dict(
        title='Mean of relative wealth index',
        source='geohub:/api/datasets/fcde6ab53a79657a27906b3248a1979d',
        operator="mean"
    )
    variables['rwi_min'] = dict(
        title='Minimum of relative wealth index',
        source='geohub:/api/datasets/fcde6ab53a79657a27906b3248a1979d',
        operator="min"
    )
    variables['rwi_max'] = dict(
        title='Maximum of relative wealth index',
        source='geohub:/api/datasets/fcde6ab53a79657a27906b3248a1979d',
        operator="max"
    )
    return variables