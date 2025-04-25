from rapida.session import Session
from rapida.util.http_get_json import http_get_json
import httpx
import logging

logger = logging.getLogger(__name__)

def resolve_geohub_url(dataset_url: str, link_name: str = None):
    """
    Resolve geohub dataset URL

    It should follow the following format:
        - `geohub:/api/datasets/{dataset_id}` (recommended)
        - `https:{geohub_hostname}/api/datasets/{dataset_id}` (not recommended)

    if datasset_url is not following the formats, it returns provided dataset_url back.

    If link_name is not provided or a provided link_name does not exist, it returns `dataset.properties.url`.

    Prior to use this function, it requires to initialize ~/.rapida/config.json by `rapida init` command.

    Usage:
        data_url = "geohub:/api/datasets/{dataset_id}"
        fgb_url = resolve_geohub_url(dataset_url=data_url, link_name="flatgeobuf")

    :param dataset_url: GeoHub dataset URL.
    :param link_name: Optional link name. e.g., `download`, `flatgeobuf` to fetch a link URL from dataset object.

    :returns: `dataset.properties.url` or a link href from dataset.properties.links
    """
    source_url = dataset_url

    if source_url:
        with Session() as ses:
            geohub_endpoint = ses.get_config_value_by_key('geohub_endpoint')
            if geohub_endpoint is None:
                raise RuntimeError(
                    "Tool initialization is not likely done properly. Please execute 'rapida init' to initialize the tool first.")
        if source_url.startswith('geohub:'):
                source_url = source_url.replace('geohub:', geohub_endpoint)
        elif source_url.startswith(geohub_endpoint):
            pass
        else:
            logger.info(f"Unsupported dataset URL: {source_url}. It returns original dataset URL provided.")
            return dataset_url

    try:
        timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
        data = http_get_json(url=source_url, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f'Failed to get electricity grid from  {dataset_url}. {e}')

    blob_url = None
    if link_name:
        # if link_name exists under properties.links, return the URL from it.
        # if not exist, return properties.url
        for link in data['properties']['links']:
            if link_name in link['rel']:
                blob_url = link['href']
    else:
        # otherwise return dataset.properties.url
        blob_url = data['properties']['url']

    return blob_url


# if __name__ == "__main__":
#     setup_logger(name='rapida', level=logging.INFO)
#
#     data_http_url = "https://geohub.data.undp.org/api/datasets/019a4692967f6412fb70808ee325d0e3"
#     fgb_url = resolve_geohub_url(dataset_url=data_http_url, link_name="flatgeobuf")
#     logger.info(f"fgb_url: {fgb_url}")
#
#     data_url = "geohub:/api/datasets/019a4692967f6412fb70808ee325d0e3"
#     fgb_url = resolve_geohub_url(dataset_url=data_url, link_name="flatgeobuf")
#     logger.info(f"fgb_url: {fgb_url}")
#
#     download_url = resolve_geohub_url(dataset_url=data_url, link_name="download")
#     logger.info(f"download_url: {download_url}")
#
#     url = resolve_geohub_url(dataset_url=data_url)
#     logger.info(f"url: {url}")