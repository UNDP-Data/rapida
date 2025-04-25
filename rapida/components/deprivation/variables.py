from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


def generate_variables():
    """
    Generate relative wealth index variables dict
    :return RWI variables definition
    """

    # https://www.earthdata.nasa.gov/data/catalog/sedac-ciesin-sedac-pmp-grdi-2010-2020-1.00
    license = "Creative Commons Zero (CC0)"
    attribution = "Center for International Earth Science Information Network - CIESIN - Columbia University"

    variables = OrderedDict()
    names = 'Average', 'Smallest', 'Largest'
    for i, operator in enumerate(['mean', 'min', 'max']):
        variables[f'depriv_{operator}'] = dict(
            title=f'{names[i]} value of relative deprivation index',
            source='geohub:/api/datasets/8b885646578937906c1d2759657f98e4',
            operator=operator,
            license=license,
            attribution=attribution,
        )
    return variables