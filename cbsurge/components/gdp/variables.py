from collections import OrderedDict
import logging
from cbsurge.session import Session

logger = logging.getLogger(__name__)

UNDP_AZURE_GDP_PATH = 'az:{account_name}:{stac_container_name}/gdp/{year}/gdp.tif'


def generate_variables():
    """
    Generate GDP variables dict
    :return RWI variables definition
    """

    license = "Creative Commons Attribution 4.0 International"
    attribution = "Wang, T., & Sun, F. (2023). Global gridded GDP under the historical and future scenarios"

    variables = OrderedDict()

    with Session() as session:
            source = UNDP_AZURE_GDP_PATH.replace("{account_name}", session.get_account_name()).replace("{stac_container_name}", session.get_stac_container_name())

            for operator in ['sum', 'mean', 'min', 'max']:
                percentage = True if operator == 'sum' else False
                variables[f'gdp_{operator}'] = dict(
                    title=f'GDP {operator}',
                    source=source,
                    operator=operator,
                    percentage=percentage,
                    license=license,
                    attribution=attribution,
                )
    return variables