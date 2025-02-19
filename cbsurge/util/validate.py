import logging

import httpx

logger = logging.getLogger(__name__)


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
