import json
import os.path
from rapida.ntl import cache
from rapida.ntl.nasa.const import (
    STREAM2CATALOG,
    COLLECTIONS,
    CMR_STAC_ROOT,
    OPERATIONAL, ARCHIVE,
    NTL_FILENAME_PATTERN
)
from rapida.ntl.nasa.util import timestamp_format
import math
from datetime import datetime, timedelta, date
import logging
from pystac_client import Client
from rich.progress import Progress

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


def stac_search(stream:str, processing_level:str, dt:datetime, bbox:tuple[float]):

    catalog_name = STREAM2CATALOG[stream]
    catalog_collections = COLLECTIONS[catalog_name]
    catalog_processing_levels = catalog_collections.keys()
    assert processing_level in catalog_processing_levels, (f'Invalid processing level {processing_level} for {catalog_name}. \''
                                                           f'Valid processing levels {catalog_processing_levels}')

    available_collections = sorted(catalog_collections[processing_level], reverse=True) #we preffe



    logger.info(f'Searching for {processing_level} imagery in catalog "{catalog_name}" collections: {available_collections}')
    logger.debug(f'Searching for {processing_level} imagery in catalog "{catalog_name}" collections: {available_collections} ' \
                 f'for {dt} and {bbox} geaographic area')
    stac_url = f'{CMR_STAC_ROOT}{catalog_name}'
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
                    _, file_name = os.path.split(url)
                    match = NTL_FILENAME_PATTERN.match(file_name)
                    meta = match.groupdict()
                    product = meta['product']
                    dt = datetime.strptime(f'{meta["year"]}{meta["doy"]}', '%Y%j')
                    tile = meta['tile']
                    timestamp = dt.strftime(timestamp_format(product_id=product))
                    key = f'{product}_{timestamp}'
                    cache.store(key=key, url=url, tile=tile)
                    urls.append((product,timestamp, tile, url))

        return urls


def search(
        processing_level: str,
        nominal_date: date,
        bbox: tuple[float, float, float, float],
        stream: str = None,
        max_concurrency: int = 5,

        progress: Progress = None
):
    """
    Unified high-performance orchestrator for NASA Black Marble imagery.
    Forces both NRT and STD streams to download files locally first for maximum
    processing speed, bypassing slow network/vsicurl VRT parsing.
    """
    minlon, minlat, maxlon, maxlat = bbox
    now = datetime.now()
    plevel = processing_level.upper()
    # --- 1. Harmonized Temporal Logic ---
    if stream == OPERATIONAL:
        days_difference = abs((now - nominal_date).days)
        if days_difference > 7:
            raise ValueError(f'Invalid target_date={nominal_date}.{stream} stream holds max 7 days of data. ')

    if 'A1' in plevel or 'A2' in plevel:
        # A2 and historical/standard A1 target noon dead-center
        midlat = (minlat + maxlat) * 0.5
        midlon = (minlon + maxlon) * 0.5
        utc_offset_hours = midlon / 15.0
        local_overpass_utc = nominal_date + timedelta(hours=1.5) - timedelta(hours=utc_offset_hours)
        local_overpass_doy = local_overpass_utc.strftime('%j')
        start_dt = local_overpass_utc - timedelta(minutes=45)
        end_dt = local_overpass_utc + timedelta(minutes=45)
        dt = [start_dt, end_dt]
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


    # --- 2. Catalog Search ---
    if progress:
        progress_task = progress.add_task(
            description=f'[red]Searching {processing_level} ({stream}) imagery for {nominal_date.date()}',
            total=None
        )

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