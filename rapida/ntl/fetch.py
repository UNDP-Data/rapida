
from datetime import datetime
import numbers
import logging
from rich.progress import Progress
from rapida.components.ntl.variables import generate_variables
from rapida.ntl.utils import get_intersecting_tiles
from rapida.ntl.nasa.search import search
from rapida.ntl.nasa import const as nasaconst
from rapida.ntl.noaa.search import async_search_granules
from rapida.ntl.noaa.cmask import select_required_granules
from rapida.ntl.nasa.io import download as download_from_nasa
from rapida.ntl.noaa.io import download as download_from_noaa
import asyncio
DELIVERABLES = tuple([g.upper() for g in generate_variables()])

logger = logging.getLogger('rapida')

async def download_and_track(granule, dest_dir, prog_bar):
    # Run the actual download
    results = await download_from_noaa(
        satellite=granule.sat,
        timestamp=granule.timestamp,
        dst_dir=dest_dir,
        progress=prog_bar
    )
    result_dict = {}
    # Return both the timestamp AND the result
    for e in results:
        result_dict[e[0]] = e[1]
    return granule.timestamp, result_dict

async def fetch(bbox:tuple[numbers.Number]=None, nominal_date:datetime=None,
                deliverable:str=None, dst_dir:str=None, progress:Progress=None):
    """
    Indentify and download the BEST available data suitable to detect outages.
    :param dst_dir:
    :param bbox:
    :param nominal_date:
    :param progress:
    :param deliverable
    :return:
    """
    deliverable = deliverable.lower()


    if 'noaa' in deliverable: # operational real time data
        logger.info(f'Going to predict VIIRS satellite passes for {nominal_date.date()} over target area: {bbox}')
        granules = await async_search_granules(
            satellites=None, nominal_date=nominal_date, bbox=bbox,
            cmask=True, progress=progress)
        if not granules:
            logger.info(f'No descending granules for found for {nominal_date.date()} over target area {bbox}')
            return
        logger.info(f'Found {len(granules)} descending granules')
        selected_granules = select_required_granules(sorted_granules=granules, bbox=bbox, progress=progress)
        logger.info(f'Selected {len(selected_granules)}  granule(s) that cover(s) bbox {bbox}')
        tasks = []
        for granule in selected_granules:
            # We no longer need a dictionary, just a simple list of tasks
            task = asyncio.create_task(
                download_and_track(granule, dst_dir, progress)
            )
            tasks.append(task)

        downloaded_files = {}

        for coro in asyncio.as_completed(tasks, timeout=100 * 3 * len(tasks)):
            try:
                # Unpack the tuple we returned from our wrapper
                timestamp, downloaded_files_dict = await coro
                downloaded_files[timestamp] = downloaded_files_dict
                logger.info(f'Downloaded operational VIIRS images for timestamp {timestamp}')
            except Exception as e:
                logger.error(e)

            except asyncio.CancelledError as ce:
                for atask in tasks:
                    if not atask.done():
                        atask.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        return downloaded_files

    expected_tiles = get_intersecting_tiles(bbox=bbox)
    routes = nasaconst.ROUTES
    stream = nasaconst.ARCHIVE
    if deliverable == 'baseline':
        processing_levels = ['A3']
    if 'nasa' in deliverable: #data from NASA LAADS catalogs
        processing_levels = 'A2', 'A1'  # best daily data ???
        if 'nrt' in deliverable:
            stream = nasaconst.OPERATIONAL

    downloaded_files = {}

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
                f'Selected {len(selected_urls)} images for deliverable {deliverable} for processing level {processing_level} route {route}')

            timestamps = set(selected_urls.values())
            if len(timestamps) > 1:
                logger.info(
                    f'Got more than one timestamp for for {selected_product} stream {stream} and processing level {processing_level} over route {route} ')
                continue

            timestamp, *_ = timestamps
            urls = list(selected_urls)

            downloaded = await download_from_nasa(timestamp=timestamp, product=selected_product, dst_dir=dst_dir, urls=urls,
                                              progress=progress)


            logger.info(f'Successfully downloaded {len(downloaded[timestamp])} selected images ')

            if not timestamp in downloaded_files:
                downloaded_files[timestamp] = {}


            downloaded_files[timestamp].update({selected_product:downloaded[timestamp]})

            # Break out of the 'routes' loop to stop searching for this processing_level
            break


    return downloaded_files




