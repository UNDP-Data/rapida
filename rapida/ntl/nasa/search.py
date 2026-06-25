import os.path

from sympy.physics.units import minute, second

from rapida.ntl import cache
from rapida.ntl.nasa import const
from rapida.ntl.utils import timestamp_format
import math
from datetime import datetime, timedelta
import logging
from pystac_client import Client
from rich.progress import Progress
from rapida.ntl.utils import get_intersecting_tiles
import httpx
from typing import Optional
from rapida.util.http_get_json import http_get_json
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


def url2result(url:str, store=True):

    _, file_name = os.path.split(url)
    match = const.NTL_FILENAME_PATTERN.match(file_name)
    meta = match.groupdict()
    product = meta['product']
    dt = datetime.strptime(f'{meta["year"]}{meta["doy"]}', '%Y%j')
    tile = meta['tile']
    timestamp = dt.strftime(timestamp_format(product_id=product))
    key = f'{product}_{timestamp}'
    if store:
        cache.store(key=key, value=url, tile=tile)
    return product, timestamp, tile, url



def calculate_local_utc_old(stream:str, processing_level:str, nominal_date: datetime, bbox:tuple[float, float, float, float],
                        route:str=None, products:list[str] =None):
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

        if route == 'STAC':
            day=15
            # A3 Monthly composites target mid-month. If current month, step back one month.
            if now.month == nominal_date.month and now.year == nominal_date.year:
                dt = nominal_date.replace(month=now.month-1, day=day)
            else:
                dt = nominal_date.replace(day=day)
        else:
            months_back = 3
            day = 1
            for m in range(1, months_back+1):
                exists = True
                for product in products:
                    dt = nominal_date.replace(month=now.month - m, day=day)
                    content_url = f'{const.API_CONTENT[stream]}/{product}/{dt.strftime("%Y/%j")}'
                    try:
                        content = http_get_json(url=content_url, timeout=10)
                        exists &= content is not None
                    except httpx.HTTPStatusError:
                        exists = False
                if exists:break
    elif 'A4' in plevel:
        if now.year > nominal_date.year:
            raise ValueError(f'Can not search in future! Please adjust target date')
        month = 7 if route == 'STAC' else 1 # A4 Annual composites target July 1st on stac and jan 1st on api
        if now.year == nominal_date.year:
            dt = nominal_date.replace(year=now.year - 1, month=month, day=1)
        else:
            dt = nominal_date.replace(month=month, day=1)
    else:
        raise ValueError(f'Invalid stream {stream} for NASA NTL data')

    return dt


def stac_catalog_has_data(stream: str, processing_level: str, dt: datetime,
                          bbox: tuple[float, float, float, float], products: list[str] = None) -> bool:
    """
    Pure 1:1 clone of stac_search logic. Hardened to expose local import errors.
    """
    catalog_name = const.STREAM2CATALOG[stream]
    catalog_collections = const.COLLECTIONS[catalog_name]

    if not products:
        available_collections = sorted(catalog_collections[processing_level.upper()], reverse=True)
    else:
        available_collections = list(products)

    stac_url = f'{const.CMR_STAC_ROOT}{catalog_name}'

    try:
        # If Client is not imported, this will instantly throw a NameError
        catalog = Client.open(url=stac_url)

        search_result = catalog.search(
            collections=available_collections,
            datetime=dt,  # Exact match, identical to your stac_search
            bbox=bbox
        )

        if search_result.matched():
            expected_tiles = get_intersecting_tiles(bbox=bbox)
            items = search_result.item_collection()

            for itm in items:
                for asset_key, asset in itm.assets.items():
                    if asset.href.endswith('.h5') and asset.href.startswith(('https', 'http')):
                        _, file_name = os.path.split(asset.href)
                        match = const.NTL_FILENAME_PATTERN.match(file_name)
                        if match:
                            meta = match.groupdict()
                            if meta['tile'] in expected_tiles:
                                return True

        return False

    except Exception as e:
        # ELEVATED TO ERROR: This will print the exact Python crash to your console
        logger.error(f"CRITICAL STAC PROBE FAILURE for {dt.strftime('%Y-%m')}: {e}", exc_info=True)
        return False


def calculate_local_utc(stream: str, processing_level: str, nominal_date: datetime,
                        bbox: tuple[float, float, float, float],
                        route: str = None, products: list[str] = None):
    """
    Calculate VIIRS satellites local overpass time in UTC TZ
    """
    minlon, minlat, maxlon, maxlat = bbox
    now = datetime.now()
    plevel = processing_level.upper()

    if stream == const.OPERATIONAL:
        days_difference = abs((now - nominal_date).days)
        if days_difference > 7:
            raise ValueError(f'Invalid target_date={nominal_date}. {stream} stream holds max 7 days of data.')

    if 'A1' in plevel or 'A2' in plevel:
        midlon = (minlon + maxlon) * 0.5
        utc_offset_hours = midlon / 15.0
        local_overpass_utc = nominal_date + timedelta(hours=1.5) - timedelta(hours=utc_offset_hours)
        dt = local_overpass_utc

    elif 'A3' in plevel:
        months_back = 3
        day = 15 if route == 'STAC' else 1
        current_probe_date = nominal_date

        if now.year == current_probe_date.year and now.month == current_probe_date.month:
            current_probe_date = current_probe_date.replace(day=1) - timedelta(days=1)

        found_valid_dt = False

        for attempt in range(months_back + 1):
            dt = current_probe_date.replace(day=day)
            exists = True

            if route == 'API':
                for product in products:
                    content_url = f'{const.API_CONTENT[stream]}/{product}/{dt.strftime("%Y/%j")}'
                    try:
                        content = http_get_json(url=content_url, timeout=10.0)
                        if not content or len(content) == 0:
                            exists = False
                            break
                    except Exception as e:
                        logger.debug(f"API Probe failed for {product} on {dt.strftime('%Y-%m')}: {e}")
                        exists = False
                        break

            elif route == 'STAC':
                # Symmetrical STAC Probe using the dynamic URL & spatial filter helper
                exists = stac_catalog_has_data(
                    stream=stream,
                    processing_level=plevel,
                    dt=dt,
                    bbox=bbox,
                    products=None  # Passing None triggers the automatic const fallback just like stac_search
                )

            if exists:
                found_valid_dt = True
                break
            else:
                logger.debug(f"No {route} data found for {current_probe_date.strftime('%Y-%m')}. Stepping back.")
                current_probe_date = current_probe_date.replace(day=1) - timedelta(days=1)

        if not found_valid_dt:
            logger.warning(
                f"Catastrophic latency: Could not find published A3 data via {route} within {months_back} months of {nominal_date.strftime('%Y-%m')}."
            )
            dt = nominal_date.replace(day=day)

    elif 'A4' in plevel:
        if nominal_date > now:
            raise ValueError('Cannot search in the future! Please adjust target date.')

        month = 7 if route == 'STAC' else 1

        if now.year == nominal_date.year:
            dt = nominal_date.replace(year=nominal_date.year - 1, month=month, day=1)
        else:
            dt = nominal_date.replace(month=month, day=1)

    else:
        raise ValueError(f'Invalid stream {stream} for NASA NTL data')

    return dt

def api_search(stream:str, products:str, dt:datetime, bbox:tuple[float, float, float, float], push_to_cache:bool=True)-> list[str]:
    tiles = get_intersecting_tiles(bbox=bbox)
    urls = []
    logger.info(
        f'Searching for imagery in products "{products}')
    for product in products:
        content_url = f'{const.API_CONTENT[stream]}/{product}/{dt.strftime("%Y/%j")}'
        with httpx.Client() as client:
            # Fetch the JSON directory listing
            response = client.get(content_url, timeout=10.0)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                continue
            # MODAPS returns a flat JSON array of file objects
            payload = response.json()['content']
            for item in payload:
                file_name = item.get("name", "")
                for tile in tiles:
                    if tile in file_name:
                        url = item.get('downloadsLink')
                        result = url2result(url=url, store=push_to_cache)
                        urls.append(result)


    return urls

def stac_search(stream:str=None, processing_level:Optional[str]=None, products:list[str]=None, dt:datetime=None,
                bbox:tuple[float, float, float, float]=None, push_to_cache:bool=True):

    catalog_name = const.STREAM2CATALOG[stream]
    catalog_collections = const.COLLECTIONS[catalog_name]
    catalog_processing_levels = catalog_collections.keys()

    try:
        s,e = dt
        s = s.replace(hour=0, minute=0, second=0)
        e = e.replace(hour=23, minute=59, second=59)
        dt = [dt.isoformat() + "Z" for dt in (s, e)]

    except TypeError:
        pass
    if not products:

        available_collections = sorted(catalog_collections[processing_level], reverse=True)

        assert processing_level in catalog_processing_levels, (f'Invalid processing level {processing_level} for {catalog_name}. \''
                                                           f'Valid processing levels {catalog_processing_levels}')
    else:
        available_collections = list(products)
    logger.info(
        f'Searching for imagery in catalog "{catalog_name}" collections: {available_collections}')




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
        expected_tiles = get_intersecting_tiles(bbox=bbox)
        items = search_result.item_collection()

        for itm in items:
            #print(json.dumps(itm.to_dict(), indent=4))
            for asset_key, asset in itm.assets.items():
                # Look for the .h5 file, but specifically grab the HTTPS link
                if asset.href.endswith('.h5') and asset.href.startswith(('https', 'http')):
                    url = asset.href
                    _, file_name = os.path.split(url)
                    match = const.NTL_FILENAME_PATTERN.match(file_name)
                    meta = match.groupdict()
                    tile = meta['tile']
                    if not tile in expected_tiles:
                        continue
                    result = url2result(url=url, store=push_to_cache)
                    urls.append(result)

        return urls


def search(
        processing_level: str,
        nominal_date: datetime,
        bbox: tuple[float, float, float, float],
        stream: str = None,
        route:str = None,
        push_to_cache:bool=False,
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
    products = stream_products[processing_level]
    dt = calculate_local_utc(stream=stream,processing_level=processing_level,
                             nominal_date=nominal_date, bbox=bbox, route=route, products=products)
    expected_tiles = get_intersecting_tiles(bbox=bbox)
    expected_tiles_count = len(expected_tiles)
    cached_results = []
    expected_products_count = len(products)
    found_products_count = 0
    found_tiles_count = 0
    keys = []
    for product in products:
        timestamp = dt.strftime(timestamp_format(product_id=product))
        key = f'{product}_{timestamp}'
        for tile in expected_tiles:
            url = cache.fetch(key=key, tile=tile)
            if url:
                cached_results.append(url2result(url=url, store=False))
                found_tiles_count += 1
        found_products_count += 1
        keys.append(key)

    # Only short-circuit if the cache successfully returned expected number of tiles for EVERY product requested
    if found_products_count == expected_products_count and found_tiles_count >= expected_tiles_count*expected_products_count:
        logger.info(f"Full cache hit for {keys}. Bypassing network search.")
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
            push_to_cache=push_to_cache
        )
    else:
        urls = stac_search(
            stream=stream,
            processing_level=processing_level,
            dt=dt,
            bbox=bbox,
            push_to_cache=push_to_cache
        )

    if not urls:
        logger.info(f"No imagery found for stream {stream} route {route} level {processing_level} on {nominal_date.date()}")
        return
    else:
        if progress and 'progress_task' in locals():
            progress.update(progress_task,
                            description=f'[green]✅ Found {len(urls)} tiles for {processing_level} ({stream})',
                            completed=len(urls),
                            total=len(urls))

        return urls