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
        source='geohub:/api/datasets/019a4692967f6412fb70808ee325d0e3',
        operator="mean"
    )
    return variables