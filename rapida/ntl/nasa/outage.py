import os
from osgeo import gdal
import fsspec
import h5py
import secrets
from datetime import datetime, timedelta
import numbers
import logging
from rich.progress import Progress
from rapida.components.ntl.variables import generate_variables
from rapida.ntl.fetch import fetch
from scipy.ndimage import uniform_filter
import numpy as np
from rapida.ntl.nasa import const as nasaconst
from rapida.ntl.nasa.io import download
import rasterio
from scipy.ndimage import label
from rapida.util.geo import gdal_callback
from rapida.ntl import vis
import urllib

logger = logging.getLogger('rapida')


def build_viirs_vrt(
        paths: list[str],
        sds_name: str,
        is_remote: bool = False,
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
        tile_vrts += [vrt_path, header_file_path]

    else:
        tile_vrts += [vrt_path]
    return tile_vrts


async def download_and_extract(file_urls: list[str] = None, stream: str = None, route=None, processing_level=None,
                               product: str = None,
                               bbox: tuple[float, float, float, float] = None, vsimem=False, dst_dir: str = None,
                               progress=None,
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


def spatial_filter(outage_map, min_size=2):
    # 1. Group connected pixels into "clumps"
    labeled_array, num_features = label(outage_map)

    # 2. Count how many pixels are in each clump
    clump_sizes = np.bincount(labeled_array.ravel())

    # 3. Create a mask of clumps that meet your size requirement
    mask_size = clump_sizes >= min_size

    # 4. Filter the original map (clump 0 is the background, so we ignore it)
    mask_size[0] = 0
    return mask_size[labeled_array]


def pure_numpy_ssim(nrt_masked: np.ma.MaskedArray, baseline_masked: np.ma.MaskedArray, win_size: int = 7) -> np.ndarray:
    """
    Computes the Structural Similarity Index (SSIM) between two images
    using purely NumPy and SciPy uniform filters.

    Returns a 2D NumPy array matching the input dimensions (values from -1 to 1).
    """
    # 1. Fill masked regions with 0.0 so they don't break the rolling windows
    img1 = nrt_masked.filled(0.0)
    img2 = baseline_masked.filled(0.0)

    # 2. Dynamic range stability constants (C1, C2)
    # Based on log1p values, data max is roughly 9.0 to 11.0
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 3. Compute local means (mu) using a uniform box filter
    mu1 = uniform_filter(img1, size=win_size, mode='reflect')
    mu2 = uniform_filter(img2, size=win_size, mode='reflect')

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 4. Compute local variances (sigma_sq) and covariance (sigma_12)
    # Formula: Var(X) = E[X^2] - (E[X])^2
    sigma1_sq = uniform_filter(img1 ** 2, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=win_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=win_size) - mu1_mu2

    # 5. Compute full SSIM map
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den

    return ssim_map



async def detect_outage(bbox:tuple[numbers.Number]=None, nominal_date:datetime=None, deliverable:str=None,
                dst_dir:str=None, progress:Progress=None):

    logger.info(f'Fetching best imagery for {bbox} {nominal_date}')
    data_urls = await fetch(bbox=bbox, nominal_date=nominal_date, progress=progress, deliverable=deliverable)

    logger.info(f'Fetching baseline imagery for {bbox} {nominal_date}')
    baseline_urls = await fetch(bbox=bbox, nominal_date=nominal_date, progress=progress, deliverable='baseline')

    # data = {}
    # with rasterio.open(target_data) as target, rasterio.open(baseline_data) as base:
    #     nrt = np.log1p(target.read(1, masked=True))
    #     baseline = np.log1p(base.read(1, masked=True))
    #     ssim = pure_numpy_ssim(nrt, baseline, win_size=7)
    #     anomaly_mask = (ssim < 0.4) & (baseline.filled(0.0) > 1.5) & (~np.ma.getmaskarray(nrt))
    #     clusters = spatial_filter(anomaly_mask, min_size=5)
    #     data[target_data] = nrt
    #     data[baseline_data] = baseline
    #     data['ssim'] = ssim
    #     data['anom'] = clusters
    #     vis.display1(data=data, title=f'All relevant processing levels imagery for {bbox} on {nominal_date.date()}')

