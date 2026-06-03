
from datetime import datetime
import numbers
import logging
from rich.progress import Progress
from rapida.components.ntl.variables import generate_variables
from rapida.ntl.nasa.util import get_intersecting_tiles
from rapida.ntl.nasa.search import search
from rapida.ntl.nasa import const as nasaconst
from rapida.ntl.noaa.search import async_search_granules, VIIRSNavigator

DELIVERABLES = tuple([g.upper() for g in generate_variables() if not 'nrt' in g])

logger = logging.getLogger('rapida')



async def fetch(bbox:tuple[numbers.Number]=None, nominal_date:datetime=None, deliverable:str=None,
                progress:Progress=None):
    """
    Indentify and download the BEST quality available data suitable to detect outages.
    :param bbox:
    :param nominal_date:
    :param progress:
    :param deliverable
    :return:
    """
    deliverable = deliverable.lower()


    if 'noaa' in deliverable: # operational real time data
        granules = await async_search_granules(
            satellites=None, nominal_date=nominal_date, bbox=bbox,
            cmask=True, progress=progress)

    expected_tiles = get_intersecting_tiles(bbox=bbox)
    routes = nasaconst.ROUTES
    stream = nasaconst.ARCHIVE
    if deliverable == 'baseline':
        processing_levels = ['A3']
    if 'nasa' in deliverable: #data from NASA LAADS catalogs
        processing_levels = 'A2', 'A1'  # best daily data ???
        if 'nrt' in deliverable:
            stream = nasaconst.OPERATIONAL

    urls = {}

    for processing_level in processing_levels:
        for route in routes:

            found_urls = search(
                processing_level=processing_level,
                nominal_date=nominal_date,
                bbox=bbox,
                stream=stream,
                route=route,
                progress=progress,
                push_to_cache=True
            )

            if not found_urls:
                logger.debug(
                    f'No data was found for deliverable {deliverable} at processing level {processing_level} through route {route}')
                continue

            products = tuple(sorted(set([e[0] for e in found_urls])))
            selected_product = products[0]

            # Simplified dictionary comprehension
            selected_urls = {e[-1]: e[1] for e in found_urls if e[0] == selected_product}

            if len(selected_urls) != len(expected_tiles):
                logger.info(
                    f'Expected to get {len(expected_tiles)} for {selected_product} stream {stream} and processing level {processing_level} over route {route}. Got {len(selected_urls)}')
                continue

            logger.info(
                f'Selected {len(selected_urls)} images for deliverable {deliverable} at processing level {processing_level} {route}')

            # Save the successful URLs
            urls[selected_product] = selected_urls

            # Break out of the 'routes' loop to stop searching for this processing_level
            break

    return urls

    # for product, file_urls in all_urls.items():
    #
    #
    #
    #
    #     return await download_and_extract(urls=urls, stream=stream, route=route, processing_level=processing_level, expected_tiles=expected_tiles,
    #                                           bbox=bbox, vsimem=vsimem, deliverable=deliverable, dst_dir=dst_dir, progress=progress)



