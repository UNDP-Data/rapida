from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


def generate_variables():
    """
    Generate relative wealth index variables dict
    :return RWI variables definition
    """

    # https://www.earthdata.nasa.gov/data/catalog/sedac-ciesin-sedac-pmp-grdi-2010-2020-1.00
    license = "Creative Commons Zero (CC0 1.0)"
    attribution = "NASA Black Marble data courtesy of the NASA Goddard Space Flight Center’s Terrestrial Information Systems Laboratory and the Earth from Space Institute (EfSI)."

    variables = OrderedDict()
    variables['noaa_nrt_outage'] = dict(title='Outage detected through NOAA real time data',
                                   source=f"NOAA",
                                   operator='sum',
                                   percentage=True,
                                   license=license,
                                   attribution="Data sourced from the NOAA Open Data Dissemination (NODD) Program, utilizing the Joint Polar Satellite System (JPSS) VIIRS Sensor Data Records hosted on [AWS Open Data / Google Cloud Public Datasets].",
                                   )
    variables['nasa_oper_outage'] = dict(title='Outage detected through NASA Black Marble operational (LANCEMODIS) data',
                                    source=f"NASA",
                                    operator='sum',
                                    percentage=True,
                                    license=license,
                                    attribution=attribution,
                                    )

    variables['nasa_arch_outage'] = dict(title='Outage detected through NASA Black Marble archived (LAADS) data',
                                         source=f"NASA",
                                         operator='sum',
                                         percentage=True,
                                         license=license,
                                         attribution=attribution,
                                         )

    return variables