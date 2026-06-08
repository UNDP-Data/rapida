from pathlib import Path
import httpx
from osgeo import gdal
import fsspec
import h5py
import secrets
import asyncio
import os
from rich.progress import Progress
from rapida.ntl import cache
from rapida.util.download_remote_file import download_remote_files
import logging
from urllib.parse import urlparse
from rapida.ntl.nasa.search import stac_search
from rapida.ntl.utils import get_intersecting_tiles
from datetime import datetime
import numbers
from rapida.ntl.nasa import const as nasaconst
from rapida.util.geo import gdal_callback
import urllib

logger = logging.getLogger(__name__)



def h52vrt(
        paths: list[str],
        sds_name: str,
        is_remote: bool = False,
        vrt_path: str = None,
        bbox: tuple[float, float, float, float] = None
) -> str:
    """
    Creates an in-memory/local disk mosaicked VRT from an iterable of local or remote VIIRS HDF5 files.
    Bypasses GDAL's internal georeferencing failures by building explicit XML.
    """
    if not paths:
        raise ValueError("The paths iterable cannot be empty.")

    # 1. Setup Disk-less Auth (RAM only)
    storage_options = {}
    header_file_path = "/vsimem/gdal_auth.txt"

    if is_remote:
        token = os.environ.get('EARTHDATA_TOKEN')
        # Add this sanity check
        if not token:
            raise ValueError("CRITICAL: EARTHDATA_TOKEN environment variable is not set or is empty!")
        storage_options = {
            "block_size": 1024 * 1024,
            "headers": {"Authorization": f"Bearer {token}"}
        }
        # Inject the header file directly into GDAL's virtual memory
        gdal.FileFromMemBuffer(header_file_path, f"Authorization: Bearer {token}\r\n".encode('utf-8'))

    hdf_root = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data_Fields"
    dataset_path = f"{hdf_root}/{sds_name}"

    tile_vrts = []
    global_metadata = {}
    # 2. Process each path to create geographically aware sub-VRTs
    for i, path in enumerate(paths):
        # Use fsspec to natively handle both local OS files and HTTP streams
        with fsspec.open(path, "rb", **(storage_options if is_remote else {})) as f:
            with h5py.File(f, 'r') as hfile:

                # Extract Extent (Required for EVERY tile to mosaic correctly)
                def get_val(key):
                    v = hfile.attrs[key]
                    return float(v[0]) if isinstance(v, (list, tuple)) or hasattr(v, '__iter__') else float(v)

                west = get_val('WestBoundingCoord')
                north = get_val('NorthBoundingCoord')
                east = get_val('EastBoundingCoord')
                south = get_val('SouthBoundingCoord')

                # Extract Dimension & Metadata ONLY from the FIRST file
                if i == 0:
                    try:
                        target_ds = hfile[dataset_path]
                    except KeyError as ke:
                        target_ds = hfile[dataset_path.replace('Data_Fields', 'Data Fields')]

                    height, width = target_ds.shape[-2:]

                    for k, v in target_ds.attrs.items():
                        if isinstance(v, bytes):
                            v = v.decode('utf-8')
                        elif hasattr(v, '__iter__') and not isinstance(v, str):
                            v = v[0].decode('utf-8') if isinstance(v[0], bytes) else str(v[0])
                        global_metadata[k] = str(v)

                    nodata_value = float(global_metadata.get('_FillValue', -999.9))

        # Calculate Geotransform for THIS specific tile
        px_w = (east - west) / width
        px_h = (south - north) / height

        # Construct GDAL connection string
        if is_remote:
            encoded = urllib.parse.quote(path, safe="")
            auth_str = f"&amp;header_file={header_file_path}" if token else ""
            vsi = f"/vsicurl?empty_dir=yes{auth_str}&amp;url={encoded}"
            conn_str = f'HDF5:"{vsi}"://{dataset_path}'
        else:
            abs_path = os.path.abspath(path)
            conn_str = f'HDF5:"{abs_path}"://{dataset_path}'
        _, file_name = os.path.split(path)

        # XML Template (EPSG:4326 is the immutable standard for VIIRS L3/L4)
        tile_xml = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
          <SRS dataAxisToSRSAxisMapping="2,1">EPSG:4326</SRS>
          <GeoTransform>{west}, {px_w}, 0.0, {north}, 0.0, {px_h}</GeoTransform>
          <VRTRasterBand dataType="Float32" band="1">
            <NoDataValue>{nodata_value}</NoDataValue>
            <SimpleSource>
              <SourceFilename relativeToVRT="0">{conn_str}</SourceFilename>
              <SourceBand>1</SourceBand>
              <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
              <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
            </SimpleSource>
          </VRTRasterBand>
        </VRTDataset>"""

        # Save individual tile VRT to memory
        tile_vrt_path = f"/vsimem/{file_name}{secrets.token_hex(6)}.vrt"
        gdal.FileFromMemBuffer(tile_vrt_path, tile_xml.encode('utf-8'))
        tile_vrts.append(tile_vrt_path)

    # 3. Mosaic all virtual tiles into the final master VRT
    vrt_opts = gdal.BuildVRTOptions(
        outputBounds=bbox,
        outputSRS='EPSG:4326',
        resolution='highest',
        resampleAlg=gdal.GRA_NearestNeighbour,
        srcNodata=nodata_value,
        VRTNodata=nodata_value
    )

    master_ds = gdal.BuildVRT(vrt_path, tile_vrts, options=vrt_opts)
    if master_ds is None:
        raise RuntimeError("GDAL failed to build the master VRT.")

    # 4. Inject the unified metadata & explicitly set NoData
    band = master_ds.GetRasterBand(1)
    band.SetMetadata(global_metadata)
    band.SetNoDataValue(nodata_value)

    master_ds.FlushCache()
    master_ds = None

    if is_remote:
        tile_vrts += [vrt_path, header_file_path]

    else:
        tile_vrts += [vrt_path]
    return tile_vrts


def extract_bb(image_files: list[str] = None, sds_name:str=None, return_gt=False,
                     bbox: tuple[float, float, float, float] = None,  progress=None
                               ):
    vrt_path = f'/vsimem/{secrets.token_hex(10)}.vrt'
    to_unlink = []
    to_unlink += h52vrt(paths=image_files, sds_name=sds_name, is_remote=False, bbox=bbox,vrt_path=vrt_path)
    to_unlink = [e for e in to_unlink if gdal.VSIStatL(e) is not None]
    translate_options = dict(
        format="MEM",
        outputSRS='EPSG:4326',
    )

    if progress:
        task = progress.add_task(f'[red]Extracting data using GDAL')
        callback_dict = dict(
            callback=gdal_callback,
            callback_data=(progress, task, None)
        )
        translate_options.update(callback_dict)

    # Pass the dataset object directly instead of the string path
    ds = gdal.Translate(destName='', srcDS=vrt_path, **translate_options)

    gt = ds.GetGeoTransform()
    [gdal.Unlink(e) for e in to_unlink]
    array = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    if return_gt:
        return array, gt
    return array

async def extract(image_files: list[str] = None, sds_name:str=None, product:str=None, dst_dir:str=None,
                  nominal_date:datetime=None, bbox: tuple[float, float, float, float] = None,  progress=None
                               ):
    product_tif_path = os.path.join(dst_dir, f'{product}_{nominal_date:%Y%m%d}_{nominal_date.timestamp()}.tif')
    _, tif_name = os.path.split(product_tif_path)

    vrt_path = f'/vsimem/{tif_name}.vrt'
    to_unlink = []
    to_unlink += h52vrt(paths=image_files, sds_name=sds_name, is_remote=False, bbox=bbox,vrt_path=vrt_path)
    to_unlink = [e for e in to_unlink if gdal.VSIStatL(e) is not None]
    translate_options = dict(
        format="GTiff",
        creationOptions=[
            "TILED=YES",  # Optimizes read performance
            "BIGTIFF=IF_SAFER",
            "COPY_SRC_MDD=YES"
        ],
        outputSRS='EPSG:4326',

    )

    if progress:
        task = progress.add_task(f'[red]Extracting data using GDAL')
        callback_dict = dict(
            callback=gdal_callback,
            callback_data=(progress, task, None)
        )
        translate_options.update(callback_dict)

    # Pass the dataset object directly instead of the string path
    ds = gdal.Translate(destName=product_tif_path, srcDS=vrt_path, **translate_options)
    ds = None

    [gdal.Unlink(e) for e in to_unlink]
    return Path(product_tif_path)

async def download_and_extract(file_urls: list[str] = None, stream: str = None, route=None, processing_level=None,
                               product: str = None, bbox: tuple[float, float, float, float] = None, vsimem=False,
                               dst_dir: str = None, progress=None,
                               ):
    timestamps = set(file_urls.values())
    if len(timestamps) > 1:
        logger.info(
            f'Got more than one timestamp for for {product} stream {stream} and processing level {processing_level} over route {route} ')
        return
    timestamp, = timestamps
    urls = list(file_urls)
    sds_name = nasaconst.SUB_DATASETS.get(processing_level)
    if not sds_name:
        raise ValueError(f"Processing level '{processing_level}' not found in mapping.")
    vrt_path = f'/vsimem/{product}_{timestamp}_{secrets.token_hex(6)}.vrt'
    to_unlink = []
    if not vsimem:
        downloaded_files = await download(timestamp=timestamp, product=product, dst_dir=dst_dir, urls=urls,
                                          progress=progress)
        to_unlink += h52vrt(paths=downloaded_files, sds_name=sds_name, is_remote=False, bbox=bbox,
                            vrt_path=vrt_path)
    else:
        to_unlink += h52vrt(paths=urls, sds_name=sds_name, is_remote=True, bbox=bbox, vrt_path=vrt_path)

    to_unlink = [e for e in to_unlink if gdal.VSIStatL(e) is not None]
    _, vrt_name = os.path.split(vrt_path)
    output_tif_path: str = os.path.join(dst_dir, vrt_name.replace('.vrt', '.tif'))

    translate_options = dict(
        format="GTiff",
        creationOptions=[
            "TILED=YES",  # Optimizes read performance
            "BIGTIFF=IF_SAFER",
            "COPY_SRC_MDD=YES"
        ],
        outputSRS='EPSG:4326',

    )

    if progress:
        task = progress.add_task(f'[red]Extracting data using GDAL')
        callback_dict = dict(
            callback=gdal_callback,
            callback_data=(progress, task, None)
        )
        translate_options.update(callback_dict)

    # Pass the dataset object directly instead of the string path
    ds = gdal.Translate(destName=output_tif_path, srcDS=vrt_path, **translate_options)
    ds = None

    [gdal.Unlink(e) for e in to_unlink]
    return output_tif_path



async def download(timestamp: str = None, product: str = None, tile:str=None,
                   bbox:tuple[float, float, float, float]=None,dst_dir:str=None, urls:list[str]=[], progress:Progress=None):

    assert timestamp not in [None, ''], f'Invalid timestamp={timestamp}'
    assert product not in [None, ''], f'Invalid product={product}'


    if not urls:
        if tile is None:
            assert bbox, f'Invalid bbox={bbox}'
        if tile is None and bbox is not None:
            tiles = get_intersecting_tiles(bbox=bbox)
        else:
            tiles = tile,
        key = f'{product.upper()}_{timestamp}'
        for tile in tiles:
            urls.append(cache.fetch(key=key, tile=tile))

        if not len(urls) == len(tiles):
            logger.info(f'Failed to locate information in {cache.CACHE_PATH} for {product}-{timestamp}-{tile or ""} \n' \
                           f'Consider searching first.')
            return

    # EarthAccess token
    ea_token = os.environ.get('EARTHDATA_TOKEN')

    # Add this sanity check
    if not ea_token:
        raise ValueError("CRITICAL: EARTHDATA_TOKEN environment variable is not set or is empty!")

    headers = {"Authorization": f"Bearer {ea_token}"}

    downloaded_files =  await download_remote_files(
        file_urls=urls,dst_folder=dst_dir, progress=progress, headers=headers
    )
    return {timestamp:downloaded_files}


async def download_tile(
        client: httpx.AsyncClient=None,
        url: str=None,
        dest_path: Path=None,
        semaphore: asyncio.Semaphore=None,
        progress: Progress=None,
        max_retries: int = 3
) -> Path:
    """
    Asynchronously downloads a single tile with exponential backoff retries.
    Uses a semaphore to prevent overwhelming the LANCE servers.
    """
    progress_task = None
    async with semaphore:
        for attempt in range(max_retries):
            try:
                # Start the streaming request
                async with client.stream("GET", url) as response:
                    response.raise_for_status()

                    # Initialize progress bar for this specific file
                    total_bytes = int(response.headers.get("Content-Length", 0))
                    if progress:
                        progress_task = progress.add_task(f'Downloading {url}', total=total_bytes,)
                    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
                    with open(tmp_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            if progress and progress_task:
                                progress.advance(progress_task, advance=len(chunk))
                    tmp_path.rename(dest_path)
                if progress and progress_task:
                    progress.update(progress_task, description=f"[green]✓ {dest_path.name}")

                return dest_path

            except httpx.HTTPError as e:
                if attempt == max_retries - 1:
                    if progress and progress_task:
                        progress.update(progress_task, description=f"[red]✗ {dest_path.name} (Failed)[/red]")
                    #progress.console.print(f"[red]Error downloading {url}: {e}[/red]")
                    return dest_path

                # Exponential backoff before retry (1s, 2s, 4s...)
                await asyncio.sleep(2 ** attempt)

            finally:
                if progress and progress_task:
                    progress.remove_task(progress_task)

async def bulk_download(bbox:tuple[numbers.Number]=None, start_date:datetime=None, end_date:datetime=None,
                           stream:str = None, products:str=None, dst_dir:str=None, progress=None):


    results = stac_search(stream=stream, products=products,
                       dt=[start_date, end_date], bbox=bbox,push_to_cache=False)


    # EarthAccess token
    ea_token = os.environ.get('EARTHDATA_TOKEN')

    # Add this sanity check
    if not ea_token:
        raise ValueError("CRITICAL: EARTHDATA_TOKEN environment variable is not set or is empty!")

    headers = {"Authorization": f"Bearer {ea_token}"}
    tasks = []
    semaphore = asyncio.Semaphore(5)
    progress_task = None
    if results:
        dest_path = Path(dst_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            for e in results:
                *r, url = e
                path = urlparse(url).path

                # 2. Get the basename from that path
                file_name = os.path.basename(path)

                filepath = dest_path / file_name

                tasks.append(asyncio.Task(
                    download_tile(client, url, filepath, semaphore), name=file_name
                ))

            if progress:
                progress_task = progress.add_task(description=f'Downloading {len(tasks)} images...', total=len(tasks))
            downloaded_files = []
            for task in asyncio.as_completed(tasks, timeout=20*len(tasks)):
                try:
                    downloaded_file = await task

                    if progress and progress_task is not None:
                        progress.update(progress_task,description=f'[green]🡇 {downloaded_file.name}', advance=1)
                    downloaded_files.append(str(downloaded_file))
                except Exception as e:
                    logger.error(e)

                except asyncio.CancelledError as ce:
                    for atask in tasks:
                        if not atask.done():
                            atask.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise