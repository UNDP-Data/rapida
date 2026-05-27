import json
import os.path
from rapida.ntl import cache
from rapida.ntl.nasa import const
from rapida.ntl.nasa.util import timestamp_format
import math
from datetime import datetime, timedelta, date
import logging
from pystac_client import Client
from rich.progress import Progress
from rapida.ntl.nasa.util import get_intersecting_tiles
import httpx
logger = logging.getLogger(__name__)






def calculate_night_hours(midlat: float, day_of_year: int) -> int:
    """
    Calculates the average hours of nighttime for a given latitude and Julian day.
    """
    # 1. Approximate Solar Declination (in radians)
    # 23.44 represents Earth's axial tilt. 81 is approx the Spring Equinox.
    tilt = math.radians(23.44)
    angle = math.radians((360 / 365.24) * (day_of_year - 81))
    declination = math.asin(math.sin(tilt) * math.sin(angle))

    # 2. Your Hour Angle math
    lat_rad = math.radians(midlat)
    cos_h = -math.tan(lat_rad) * math.tan(declination)

    # Clamp to [-1, 1] to handle Polar Day (0 night) and Polar Night (24h night)
    cos_h = max(-1.0, min(1.0, cos_h))

    # Calculate hours
    daylight_hrs = 2 * math.degrees(math.acos(cos_h)) / 15
    night_hrs = 24 - daylight_hrs

    return int(round(night_hrs))


def url2result(url:str=None, store=True):

    _, file_name = os.path.split(url)
    match = const.NTL_FILENAME_PATTERN.match(file_name)
    meta = match.groupdict()
    product = meta['product']
    dt = datetime.strptime(f'{meta["year"]}{meta["doy"]}', '%Y%j')
    tile = meta['tile']
    timestamp = dt.strftime(timestamp_format(product_id=product))
    key = f'{product}_{timestamp}'
    if store:
        cache.store(key=key, url=url, tile=tile)
    return product, timestamp, tile, url



def calculate_local_utc(stream:str, processing_level:str, nominal_date: datetime, bbox:tuple[float]):
    """
    Calculate VIIRS satellites local overpass time in UTC TZ
    :param stream:
    :param processing_level:
    :param nominal_date:
    :param bbox:
    :return:
    """
    minlon, minlat, maxlon, maxlat = bbox
    now = datetime.now()
    plevel = processing_level.upper()
    # --- 1. Harmonized Temporal Logic ---
    if stream == const.OPERATIONAL:
        days_difference = abs((now - nominal_date).days)
        if days_difference > 7:
            raise ValueError(f'Invalid target_date={nominal_date}.{stream} stream holds max 7 days of data. ')

    if 'A1' in plevel or 'A2' in plevel:
        # A2 and historical/standard A1 target noon dead-center

        midlon = (minlon + maxlon) * 0.5
        utc_offset_hours = midlon / 15.0
        local_overpass_utc = nominal_date + timedelta(hours=1.5) - timedelta(hours=utc_offset_hours)

        dt = local_overpass_utc
    elif 'A3' in plevel:
        # A3 Monthly composites target mid-month. If current month, step back one month.
        if now.month == nominal_date.month and now.year == nominal_date.year:
            prev_month = nominal_date.replace(day=1) - timedelta(days=1)
            dt = prev_month.replace(day=15)
        else:
            dt = nominal_date.replace(day=15)
    elif 'A4' in plevel:
        if now.year > nominal_date.year:
            raise ValueError(f'Can not search in future! Please adjust target date')
        if now.year == nominal_date.year:
            # A4 Annual composites target July 1st
            dt = nominal_date.replace(year=now.year - 1, month=7, day=1)
        else:
            # A4 Annual composites target July 1st
            dt = nominal_date.replace(month=7, day=1)
    else:
        raise ValueError(f'Invalid stream {stream} for NASA NTL data')
    return dt


def api_search(stream:str, products:str, dt:datetime, bbox:tuple[float])-> list[str]:
    tiles = get_intersecting_tiles(bbox=bbox)
    urls = []
    for product in products:
        content_url = f'{const.API_CONTENT[stream]}/{product}/{dt.strftime("%Y/%j")}'
        with httpx.Client() as client:
            # Fetch the JSON directory listing
            response = client.get(content_url, timeout=10.0)
            response.raise_for_status()
            # MODAPS returns a flat JSON array of file objects
            payload = response.json()['content']
            for item in payload:
                file_name = item.get("name", "")
                for tile in tiles:
                    if tile in file_name:
                        url = item.get('downloadsLink')
                        result = url2result(url=url, store=True)
                        urls.append(result)


    return urls

def stac_search(stream:str, processing_level:str, dt:datetime, bbox:tuple[float]):

    catalog_name = const.STREAM2CATALOG[stream]
    catalog_collections = const.COLLECTIONS[catalog_name]
    catalog_processing_levels = catalog_collections.keys()
    assert processing_level in catalog_processing_levels, (f'Invalid processing level {processing_level} for {catalog_name}. \''
                                                           f'Valid processing levels {catalog_processing_levels}')

    available_collections = sorted(catalog_collections[processing_level], reverse=True) #we preffe

    logger.info(f'Searching for {processing_level} imagery in catalog "{catalog_name}" collections: {available_collections}')
    logger.debug(f'Searching for {processing_level} imagery in catalog "{catalog_name}" collections: {available_collections} ' \
                 f'for {dt} and {bbox} geaographic area')
    stac_url = f'{const.CMR_STAC_ROOT}{catalog_name}'
    urls = []
    catalog = Client.open(url=stac_url)

    search_result = catalog.search(
        collections=available_collections,
        datetime=dt,
        bbox=bbox
    )

    if search_result.matched():
        logger.info(f"Found {search_result.matched()} granule(s) at {stac_url}")
        items = search_result.item_collection()

        for itm in items:
            #print(json.dumps(itm.to_dict(), indent=4))
            for asset_key, asset in itm.assets.items():
                # Look for the .h5 file, but specifically grab the HTTPS link
                if asset.href.endswith('.h5') and asset.href.startswith('https'):
                    url = asset.href
                    result = url2result(url=url, store=True)
                    urls.append(result)

        return urls


def search(
        processing_level: str,
        nominal_date: datetime,
        bbox: tuple[float, float, float, float],
        stream: str = None,
        route:str = None,
        max_concurrency: int = 5,
        progress: Progress = None
):
    """
    Unified high-performance orchestrator for NASA Black Marble imagery.
    Forces both NRT and STD streams to download files locally first for maximum
    processing speed, bypassing slow network/vsicurl VRT parsing.
    """
    stream_name = const.STREAM2CATALOG[stream]
    stream_products = const.API_PRODUCTS[stream_name]
    stream_processing_levels = stream_products.keys()
    assert processing_level.upper() in stream_processing_levels, (
        f'Invalid processing level {processing_level} for {stream_name}. \''
        f'Valid processing levels {stream_processing_levels}')

    dt = calculate_local_utc(stream=stream,processing_level=processing_level,
                             nominal_date=nominal_date, bbox=bbox)
    products = stream_products[processing_level]
    cached_results = []
    expected_products_count = len(products)
    found_products_count = 0

    for product in products:
        timestamp = dt.strftime(timestamp_format(product_id=product))
        key = f'{product}_{timestamp}'
        urls = cache.fetch(key=key)
        if urls:
            found_products_count += 1
            for url in urls:
                cached_results.append(url2result(url=url, store=False))

    # Only short-circuit if the cache successfully returned data for EVERY product requested
    if found_products_count == expected_products_count:
        logger.info("Full cache hit. Bypassing network search.")
        return cached_results
    # --- 2. Catalog Search ---
    if progress:
        progress_task = progress.add_task(
            description=f'[red]Searching {processing_level} ({stream}) imagery for {nominal_date.date()}',
            total=None
        )
    if route == 'API':
        urls = api_search(
            stream=stream,
            products=products,
            dt=dt,
            bbox=bbox,
        )
    else:
        urls = stac_search(
            stream=stream,
            processing_level=processing_level,
            dt=dt,
            bbox=bbox,
        )

    if not urls:
        logger.info(f"No imagery found for {processing_level} on {nominal_date.date()}")
        return
    else:
        if progress and 'progress_task' in locals():
            progress.update(progress_task,
                            description=f'[green]✅ Found {len(urls)} tiles for {processing_level} ({stream})',
                            completed=len(urls),
                            total=len(urls))

        return urls