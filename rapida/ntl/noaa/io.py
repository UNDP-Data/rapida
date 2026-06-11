import obstore
import os
import asyncio
from rapida.ntl.noaa.const import PRODUCTS, PRODUCT_NAMES, VIIRS_URLS,PUBLIC_CONFIG,SOURCE_NAMES
from datetime import datetime, timedelta
from typing import Iterable
import random
import logging
from rapida.ntl.noaa.cmask import bbox_in_hdf
from urllib.parse import urlparse
from rapida.ntl.noaa.const import PRODUCT2NAME
import aiofiles
from pyresample import geometry
from satpy import Scene
import numpy as np
from pathlib import Path
logger = logging.getLogger(__name__)


# The "Solid" way: Generate stores using from_url
VIIRS_STORES = {
    sat: {
        source: obstore.store.from_url(url, config=PUBLIC_CONFIG)
        for source, url in sources.items()
    }
    for sat, sources in VIIRS_URLS.items()
}


def get_viirs_stores(satellite: str):
    """
    Factory function to instantiate obstore clients on demand.
    Must only be called from inside an active asyncio event loop.
    """
    return {
        source: obstore.store.from_url(url, config=PUBLIC_CONFIG)
        for source, url in VIIRS_URLS[satellite].items()
    }

def parse_noaa_timestamp(time_str: str) -> datetime:
    """
    Converts a NOAA VIIRS string (e.g., '202604010001018') into a timezone-naive UTC datetime.
    """
    # The first 14 characters: YYYYMMDDHHMMSS
    base_time = datetime.strptime(time_str[:14], "%Y%m%d%H%M%S")

    # The 15th character: tenths of a second (1 tenth = 100,000 microseconds)
    tenths = int(time_str[14:])

    return base_time + timedelta(microseconds=tenths * 100000)


def public_url(file_path:str=None, satellite:str=None, source:str=None):

    public_cloud_url = VIIRS_URLS[satellite][source]
    parsed_public_url = urlparse(public_cloud_url)
    bucket = parsed_public_url.netloc
    if parsed_public_url.scheme == 's3':
        return f'https://{bucket}.s3.amazonaws.com/{file_path}'
    if parsed_public_url.scheme == 'gs':
        return f'https://storage.googleapis.com/{bucket}/{file_path}'


async def find_ntl(satellite: str = None, bbox: Iterable[float] = None, dt: datetime = None,
                   products: Iterable[str] = PRODUCT_NAMES, source: str = None):
    found = {}




    stores = VIIRS_STORES[satellite]

    # 1. Safely determine the primary and alternate sources
    primary_source = source if source else random.choice(SOURCE_NAMES)
    alt_source = SOURCE_NAMES[0] if primary_source == SOURCE_NAMES[1] else SOURCE_NAMES[1]

    # Calculate target times upfront to handle rollovers safely
    target_dts = [dt, dt - timedelta(minutes=1), dt + timedelta(minutes=1)]
    for product_name in products:
        product = PRODUCTS[product_name]
        sources_to_try = [primary_source, alt_source]

        # Track time patterns that we know missed the bounding box
        spatial_misses = set()
        match_found = False

        for current_source in sources_to_try:

            store = stores[current_source]
            entries_cache = {}

            for sc_dt in target_dts:
                # Format the time pattern dynamically based on the specific offset
                time_pattern = sc_dt.strftime('s%Y%m%d%H%M' if 'cloud' in product.lower() else 't%H%M')

                # Instantly skip if we already proved this timestamp misses the bbox on a previous source
                if time_pattern in spatial_misses:
                    continue

                date_path = sc_dt.strftime('/%Y/%m/%d/')
                prefix = f"{product}{date_path}"

                # Cache listed entries by prefix so we don't make duplicate network calls
                if prefix not in entries_cache:
                    entries_cache[prefix] = await obstore.list(store, prefix=prefix).collect_async()


                entries = entries_cache[prefix]

                if not entries:
                    continue

                try:
                    selected_entry = [e for e in entries if time_pattern in e['path'] and (
                                e['path'].endswith('.nc') or e['path'].endswith('.h5'))].pop()

                    file_path, file_size = selected_entry['path'], selected_entry['size']
                    public_file_url = public_url(file_path=file_path, satellite=satellite, source=current_source)
                    is_intersecting, percent = bbox_in_hdf(hdf_url=public_file_url,bbox=bbox)
                    if not is_intersecting:
                        _, file_name = os.path.split(file_path)
                        logger.info(
                            f'Skipping {file_name} from {current_source} scanned by {satellite} because it does not intersect bbox')
                        spatial_misses.add(time_pattern)
                        continue

                    if current_source not in found: #reset
                        found[current_source] = []

                    found[current_source].append((file_path, file_size, percent))
                    match_found = True
                    break

                except IndexError:
                    logger.debug(
                        f'No exact match for satellite {satellite} timestamp {time_pattern} on {store}. Considering temporal neighbors.')

                    continue

            if match_found:
                break

        if not match_found:
            logger.debug(
                f'No valid/intersecting data for product {product} and satellite {satellite} for the night {dt}')


    return found

async def locate_file(satellite:str=None, dt=None, source:str=None, products: Iterable[str] = PRODUCT_NAMES):
    found = {}

    stores = VIIRS_STORES[satellite]

    # 1. Safely determine the primary and alternate sources
    primary_source = source if source else random.choice(SOURCE_NAMES)
    alt_source = SOURCE_NAMES[0] if primary_source == SOURCE_NAMES[1] else SOURCE_NAMES[1]
    entries_cache = {}
    for product_name in products:
        #match_found = False
        product = PRODUCTS[product_name]
        sources_to_try = [primary_source, alt_source]
        time_pattern = dt.strftime('s%Y%m%d%H%M' if 'cloud' in product.lower() else 'd%Y%m%d_t%H%M')
        for current_source in sources_to_try:
            store = stores[current_source]

            date_path = dt.strftime('/%Y/%m/%d/')
            prefix = f"{product}{date_path}"
            cache_key = (current_source, prefix)
            if cache_key not in entries_cache:
                try:
                    entries_cache[cache_key] = await obstore.list(store, prefix=prefix).collect_async()
                except Exception as e:
                    logger.warning(f"Failed to list {prefix} from {current_source}: {e}")
                    entries_cache[cache_key] = []

            entries = entries_cache[cache_key]
            if not entries:
                continue
            match_gen = (
                e for e in entries
                if time_pattern in e['path'] and e['path'].lower().endswith(('.nc', '.h5'))
            )

            # next() takes the first match, or returns None if the generator is empty
            selected_entry = next(match_gen, None)

            if selected_entry:
                file_path = selected_entry['path']
                file_size = selected_entry.get('size', 0)  # Safe get
                if current_source not in found:  # reset
                    found[current_source] = []
                found[current_source].append((file_path, file_size))
                break  # Found it! Stop looking in fallback sources for this product

            else:
                logger.debug(f"Pattern {time_pattern} not found in {current_source} for {product_name}")


    return found



async def fetch_file(satellite:str=None, provider:str=None, path:str=None, size:int=None, dst_dir:str=None,
                     progress=None, progress_task = None):
    down_task = None
    try:
        adir = os.path.abspath(dst_dir)
        os.makedirs(adir, exist_ok=True)

        store = VIIRS_STORES[satellite][provider]
        rel_path, file_name = os.path.split(path)
        product = rel_path.split('/')[0]
        product_name = PRODUCT2NAME[product]
        if progress:
            down_task = progress.add_task(f'[red]Downloading  {file_name} from {provider}', total=size)
        dst_file_path = os.path.join(adir, file_name)
        response = await obstore.get_async(store, path)

        dst_path = Path(dst_file_path)
        if dst_path.exists() and dst_path.stat().st_size == response.meta['size']:
            if progress and progress_task is not None:
                progress.update(progress_task, description=f'[green]Reused {file_name} from local folder {adir}', advance=1)
            return product_name, dst_file_path, dst_path.stat().st_size
        async with aiofiles.open(dst_file_path, 'wb') as local_file:
            # The 'get' call is the async request
            async for chunk in  response.stream():
                await local_file.write(chunk)
                if progress and down_task is not None:
                    progress.update(down_task, advance=len(chunk))

        if os.stat(dst_file_path).st_size == size:
            if progress and progress_task is not None:
                progress.update(progress_task, description=f'[green]Downloaded {file_name} from {provider}', advance=1)
            return product_name, dst_file_path, size
    except Exception:

        raise
    finally:
        if progress and down_task is not None:
            progress.remove_task(down_task)


async def fetch_ntl(found_paths:dict[str, list]=None, satellite:str=None, dst_dir:str=None, progress=None):

    # Download logic (Surgical io to local SSD)
    tasks = []
    progress_task = None
    try:
        async with asyncio.TaskGroup() as tg:
            for provider, files in found_paths.items():
                if progress:
                    progress_task = progress.add_task(description=f'Downloading VIIRS images...', total=len(files))
                for path, size in files:
                    tasks.append(tg.create_task(fetch_file(
                        satellite=satellite, provider=provider,
                        path=path, size=size, progress=progress,
                        dst_dir=dst_dir, progress_task = progress_task
                    )))
    except ExceptionGroup as eg:
        for e in eg.exceptions:
            logger.error(f"❌ Sub-task failed: {e}")
    finally:

        results = [t.result() for t in tasks]
        return results


async def download(satellite:str=None, timestamp:str=None, source:str=None,
        products:Iterable[str]=PRODUCT_NAMES, dst_dir:str=None, progress=None):
    dt = datetime.strptime(timestamp, '%Y%m%d%H%M')
    logger.info(f'Locating files for satellite {satellite} timestamp {timestamp} ')
    found_files = await locate_file(satellite=satellite, dt=dt, source=source, products=products)
    return  await fetch_ntl(found_paths=found_files, dst_dir=dst_dir, satellite=satellite, progress=progress)


def bytesto(bytes, to, bsize=1024):
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    return bytes / (bsize ** a[to])





def create_area_from_geotransform(gt, array_shape):
    """
    Creates a Pyresample AreaDefinition exactly matching a GDAL array.

    Args:
        gt (tuple): GDAL GeoTransform (TopLeftX, PixelWidth, Rot, TopLeftY, Rot, PixelHeight)
        array_shape (tuple): (height, width) of the target array
    """
    height, width = array_shape

    # Calculate exact bounding box edges based on pixels
    ll_x = gt[0]
    ur_y = gt[3]
    ur_x = gt[0] + (width * gt[1])
    ll_y = gt[3] + (height * gt[5])  # gt[5] is usually negative

    area_extent = [ll_x, ll_y, ur_x, ur_y]

    # Build the EPSG:4326 definition
    area_def = geometry.AreaDefinition(
        area_id='gdal_matched_grid',
        description='Grid mapped directly from Level-3 Baseline GeoTransform',
        proj_id='EPSG:4326',
        projection={'proj': 'longlat', 'datum': 'WGS84'},
        width=width,
        height=height,
        area_extent=area_extent
    )

    return area_def





def read_and_align_sdr(sdr_path, geo_path, target_area):
    # 1. Calculate a slightly buffered crop box for Satpy
    lon_min, lat_min, lon_max, lat_max = target_area.area_extent_ll
    pad = 0.2  # Add a 0.2 degree buffer so EWA has data for edge pixels
    crop_bbox = (lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)

    # 2. Load the scene
    scn = Scene(filenames=[sdr_path, geo_path], reader='viirs_sdr')
    scn.load(['DNB'])

    # 3. Crop in memory BEFORE resampling (Massive speed/RAM boost)
    cropped_scn = scn.crop(ll_bbox=crop_bbox)

    # 4. Resample directly to the baseline's grid footprint
    resampled_scn = cropped_scn.resample(
        target_area,
        resampler='ewa',
        rows_per_scan=16,
        fill_value=-999.0  # Explicitly manage fill values
    )

    dnb_array = resampled_scn['DNB'].values
    dnb_array = np.where(dnb_array < -900, np.nan, dnb_array)

    return dnb_array * 1e5  # Scale to match Black Marble nW/cm2/sr



def read_and_align_sdr_and_cmask(sdr_path, geo_path, cmask_path, target_area):
    """
    Loads, crops, and resamples VIIRS SDR (Radiance) and EDR (Cloud Mask)
    using the native fill values extracted dynamically from the file metadata.
    """


    # 2. Load the Scenes
    sdr_scn = Scene(filenames=[sdr_path, geo_path], reader='viirs_sdr')
    sdr_scn.load(['DNB'])

    cmask_scn = Scene(filenames=[cmask_path], reader='viirs_edr')
    cmask_scn.load(['CloudMask'])


    # ---------------------------------------------------------
    # 4. DYNAMIC FILL VALUE EXTRACTION
    # Pull the exact _FillValue from the file attributes.
    # We provide a safe fallback just in case the attribute is missing.
    # ---------------------------------------------------------
    dnb_fill = sdr_scn['DNB'].attrs.get('_FillValue', np.nan)

    # EDR categorical data usually defaults to 255 or -127 for unsigned/signed bytes
    cmask_fill = cmask_scn['CloudMask'].attrs.get('_FillValue', 255)

    # 5. Resample SDR using the native fill value
    resampled_sdr = sdr_scn.resample(
        target_area,
        resampler='ewa',
        rows_per_scan=16,
        fill_value=dnb_fill
    )


    # 6. Resample Cloud Mask
    resampled_cmask = cmask_scn.resample(
        target_area,
        resampler='nearest',
        radius_of_influence=1000,
        fill_value=cmask_fill
    )

    # 7. Extract raw NumPy arrays and mask
    dnb_array = resampled_sdr['DNB'].values


    dnb_scaled = dnb_array * 1e5  # Scale to match Black Marble nW/cm2/sr
    cmask_array = resampled_cmask['CloudMask'].values

    return dnb_scaled, cmask_array