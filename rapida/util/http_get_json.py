import logging

import httpx

logger = logging.getLogger(__name__)

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