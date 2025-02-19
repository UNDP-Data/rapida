import logging

import httpx

logger = logging.getLogger(__name__)

def http_post_json(url=None, query=None, timeout=None):
    """
    Generic HTTP get function using httpx
    :param url: str, the url to be fetched
    :param timeout: httpx.Timeout instance
    :return: python dict representing the result as parsed json
    """
    assert timeout is not None, f'Invalid timeout={timeout}'
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, data={"data": query})
        response.raise_for_status()
        return response.json()