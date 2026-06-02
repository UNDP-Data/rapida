import json
import os.path
from datetime import datetime, timedelta
import numbers
import logging
from itertools import product
from rich.progress import Progress
from rapida.components.ntl.variables import generate_variables
from rapida.ntl.nasa import const as nasaconst
from rapida.ntl.nasa.util import get_intersecting_tiles
from rapida.util.geo import gdal_callback
import h5py
from rapida.ntl.nasa.search import calculate_local_utc
import fsspec
import urllib
import secrets
from rapida.ntl.nasa.io import download
from rapida.ntl.nasa.search import search
from rapida.ntl.nasa.io import download
from osgeo import gdal

DELIVERABLES = tuple([g.upper() for g in generate_variables()])
logger = logging.getLogger('rapida')


def build_viirs_vrt(
        paths: list[str],
        sds_name: str,
        is_remote:bool=False,
        vrt_path: str = None,
        bbox: tuple[float, float, float, float] = None
) -> str:
    """
    Creates an in-memory mosaicked VRT from an iterable of local or remote VIIRS HDF5 files.
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
        tile_vrts+=[vrt_path, header_file_path]

    else:
        tile_vrts+=[vrt_path]
    return tile_vrts



async def download_and_extract(urls:list[str]=None, stream:str=None, route=None, processing_level=None, expected_tiles:list[str]=None,
                               bbox: tuple[float, float, float, float] = None, vsimem=False, deliverable=None, dst_dir:str=None, progress=None):
    
    
    
    # search returns tuple product, timestamp, tile, url
    products = tuple(sorted(set([e[0] for e in urls])))  # logic HOLDS because J is before N, newer is preferred
    selected_product = products[0]
    selected_urls = dict([(e[-1], e[1]) for e in urls if e[0] == selected_product])
    if len(selected_urls) != len(expected_tiles):
        logger.info(
            f'Expected to get {len(expected_tiles)}  for {selected_product} stream {stream} and processing level {processing_level} over route {route}. Got {len(selected_urls)}')
        return 
    timestamps = set(selected_urls.values())
    if len(timestamps) > 1:
        logger.info(
            f'Got more than one timestamp for for {selected_product} stream {stream} and processing level {processing_level} over route {route} ')
        return 
    timestamp, = timestamps
    urls = list(selected_urls)
    sds_name = nasaconst.SUB_DATASETS.get(processing_level)
    if not sds_name:
        raise ValueError(f"Processing level '{processing_level}' not found in mapping.")
    vrt_path = f'/vsimem/{selected_product}_{timestamp}_{secrets.token_hex(6)}.vrt'
    to_unlink = []
    if not vsimem:
        downloaded_files = await download(timestamp=timestamp, product=selected_product, dst_dir=dst_dir, urls=urls,
                                          progress=progress)
        to_unlink += build_viirs_vrt(paths=downloaded_files, sds_name=sds_name, is_remote=False, bbox=bbox,
                                     vrt_path=vrt_path)
    else:
        to_unlink += build_viirs_vrt(paths=urls, sds_name=sds_name, is_remote=True, bbox=bbox, vrt_path=vrt_path)

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




async def fetch(bbox:tuple[numbers.Number]=None, nominal_date:datetime=None, deliverable:str=None,
                dst_dir:str=None, progress:Progress=None, vsimem=False):
    """
    Indentify and download the BEST quality available data suitable to detect outages.
    :param bbox:
    :param nominal_date:
    :param progress:
    :return:
    """
    deliverable = deliverable.lower()
    expected_tiles = get_intersecting_tiles(bbox=bbox)

    if 'noaa' in deliverable: # operational real time data
        pass
    routes = nasaconst.ROUTES
    stream = nasaconst.ARCHIVE
    if deliverable == 'baseline':
        processing_levels = ['A3']
    if 'nasa' in deliverable: #data from NASA LAADS catalogs
        processing_levels = 'A2', 'A1'  # best daily data ???
        if 'nrt' in deliverable:
            stream = nasaconst.OPERATIONAL
    for processing_level, route in product(processing_levels, routes):

        urls = search(processing_level=processing_level, nominal_date=nominal_date, bbox=bbox,
                      stream=stream, route=route,progress=progress, push_to_cache=False
                      )
        if not urls:
            logger.debug(
                f'No data was found for deliverable {deliverable} at processing level  {processing_level} through route {route}')
            continue


        return await download_and_extract(urls=urls, stream=stream, route=route, processing_level=processing_level, expected_tiles=expected_tiles,
                                              bbox=bbox, vsimem=vsimem, deliverable=deliverable, dst_dir=dst_dir, progress=progress)



