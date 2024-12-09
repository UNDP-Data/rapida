import httpx
import logging
from osgeo import gdal
import itertools

logger = logging.getLogger(__name__)

def fetch_drivers():
    d = dict()
    for i in range(gdal.GetDriverCount()):

        drv = gdal.GetDriver(i)
        d[drv.ShortName] = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
    return d

def http_get_json(url=None, timeout=None):
    """
    Generic HTTP get function using httpx
    :param url: str, the url to be fetched
    :param timeout: httpx.Timeout instance
    :return: python dict representing the result as parsed json
    """
    assert timeout is not None, f'Invalid timeout={timeout}'
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            return response.json()


def validate(url=None, timeout=10):
    """
    Generic HTTP get function using httpx
    :param url: str, the url to be fetched
    :param timeout: httpx.Timeout instance
    :return: python dict representing the result as parsed json
    """
    with httpx.Client(timeout=timeout) as client:
        response = client.head(url, timeout=timeout)
        response.raise_for_status()



def generator_length(gen):
    """
    compute the no of elems inside a generator
    :param gen:
    :return:
    """
    gen1, gen2 = itertools.tee(gen)
    length = sum(1 for _ in gen1)  # Consume the duplicate
    return length, gen2  # Return the length and the unconsumed generator