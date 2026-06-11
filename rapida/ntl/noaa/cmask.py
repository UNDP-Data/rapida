import datetime
from pyresample import geometry, kd_tree
import os.path
import urllib.parse
import h5py
import logging
import fsspec
from shapely import wkt, box, to_geojson
from typing import Iterable, Any
import concurrent
import numpy as np
from rich.progress import Progress
from osgeo import gdal
from shapely.ops import transform
from rapida.ntl import cache
from shapely.geometry import MultiPoint, Polygon
from shapely import concave_hull
from shapely.ops import unary_union
from itertools import combinations
gdal.UseExceptions()


logger = logging.getLogger(__name__)

def shift_to_360(lon, lat, z=None):
    """Transforms standard -180/180 coordinates to a 0-360 system."""
    shifted_lon = lon + 360.0 if lon < 0 else lon
    return (shifted_lon, lat) if z is None else (shifted_lon, lat, z)

def bounds_from_url(hdf_url: str):

    purl = urllib.parse.urlparse(hdf_url)
    _, filename = os.path.split(purl.path)
    storage_options = {}
    if hdf_url.startswith(('http', 'https')):
        storage_options.update({'block_size': 1024 * 1024})
    with fsspec.open(hdf_url, **storage_options) as f:
        with h5py.File(f, "r") as hfile:
            # Now it reads at HTTP speeds without the boto3 overhead
            return bounds_from_file(hfile=hfile)


def bounds_from_file(hfile, step=50) -> Polygon:
    # 1. Subsample the entire 2D array directly during the read.
    # h5py translates this slice into an optimized partial read,
    # making it very fast even over a network (fsspec).
    lat_grid = hfile['Latitude'][::step, ::step][()]
    lon_grid = hfile['Longitude'][::step, ::step][()]

    # 2. Flatten into 1D arrays
    lats = lat_grid.flatten()
    lons = lon_grid.flatten()

    # 3. Create a mask to filter out NoData/Fill values globally
    valid_mask = (lons >= -180.0) & (lons <= 180.0) & \
                 (lats >= -90.0) & (lats <= 90.0)

    valid_lons = lons[valid_mask]
    valid_lats = lats[valid_mask]

    if len(valid_lons) == 0:
        return Polygon()  # Return empty if no valid data exists

    # 4. Create a Shapely MultiPoint collection
    points = MultiPoint(np.column_stack((valid_lons, valid_lats)))

    # 5. Generate the footprint boundary
    # convex_hull creates a tight bounding polygon around the outermost points.
    poly = concave_hull(points,ratio=0.1)

    return poly.buffer(0)


def select_required_granules(sorted_granules: list, bbox: tuple, progress:Progress=None) -> list:
    boxpoly = box(*bbox, ccw=True)
    best_poly = bounds_from_url(sorted_granules[0].url)
    uncovered = boxpoly.difference(best_poly)
    if uncovered.is_empty or uncovered.area < 1e-6:
        logger.debug(f"BBOX {bbox} is covered  by first granule.")
        return [sorted_granules[0]]
    logger.info(f'bbox {bbox} is not covered completely by best swath...going to select more for complete coverage.')
    progress_task = None
    try:
        if progress:
            progress_task = progress.add_task(description=f'Selecting best granules tha cover {bbox}', total=None)
        for combo_size in range(2, len(sorted_granules) + 1):
            if progress and progress_task:
                progress.update(progress_task, description=f'Evaluating granules by pairs of {combo_size} ')
            for combo in combinations(sorted_granules, combo_size):

                # Merge the geometries for this specific combination
                merged_poly = unary_union([bounds_from_url(g.url) for g in combo])

                # 3. FLOATING-POINT SAFE COVERAGE CHECK
                # We use difference() because .within() can fail on microscopic 1e-15 gap artifacts
                uncovered = boxpoly.difference(merged_poly)

                if uncovered.is_empty or uncovered.area < 1e-6:
                    logger.debug(f"Success: BBOX covered perfectly by {combo_size} granule(s).")
                    return combo  # Exits immediately with the absolute minimum required set

        logger.warning("Exhausted all combinations. BBOX cannot be fully covered by available data.")
        return tuple()
    finally:
        if progress and progress_task:
            progress.remove_task(progress_task)

def bbox_in_hdf(hdf_url: str, bbox: Iterable[float]):
    #fs = fsspec.filesystem("http")
    purl = urllib.parse.urlparse(hdf_url)
    dst_dir, filename = os.path.split(purl.path)
    storage_options = {}
    if hdf_url.startswith(('http', 'https')):
        storage_options.update({'block_size':1024*1024})
    with fsspec.open(hdf_url, **storage_options) as f:
        with h5py.File(f, "r") as hfile:
            # Now it reads at HTTP speeds without the boto3 overhead
            #bounds_poly = wkt.loads(hfile.attrs['geospatial_bounds'].decode('utf-8'))
            bounds_poly = bounds_from_file(hfile)
            bbox_poly = box(*bbox, ccw=True)
            # with open(os.path.join('/tmp', 'bbox.geojson'), "w") as ff:
            #     ff.write(to_geojson(bbox_poly))
            # with open(os.path.join('/tmp', filename.replace('.nc', '.geojson')), "w") as f:
            #     f.write(to_geojson(bounds_poly))
            # 2. The Kiribati Ghost Detection
            minx, miny, maxx, maxy = bounds_poly.bounds
            is_idl_crosser = (maxx - minx) > 300

            if is_idl_crosser:
                # Transform both geometries to 0-360 space to fix the tear
                working_bounds = transform(shift_to_360, bounds_poly)
                working_bbox = transform(shift_to_360, bbox_poly)
            else:
                # Use standard geometries
                working_bounds = bounds_poly
                working_bbox = bbox_poly
            #if not bbox_poly.within(bounds_poly):
            if not working_bbox.intersects(working_bounds):
                return False, 0
            intersection_poly = working_bbox.intersection(working_bounds)
            perc_intersection = round(intersection_poly.area/working_bbox.area * 100)
            # with open(os.path.join('/tmp', filename.replace('.nc', '.geojson')), "w") as f:
            #     f.write(to_geojson(bounds_poly))
            return True, perc_intersection



def cloud_coverage_fast(hdf_url: str, bbox: Iterable[float],
                   lon_var: str = 'Longitude', lat_var: str = 'Latitude',
                   var_to_read: str = 'CloudMaskBinary') -> int:
    """
    Computes cloud coverage percentage matching pyresample logic identically,
    optimized to minimize HTTP chunk request thrashing.
    """
    roi_lon_min, roi_lat_min, roi_lon_max, roi_lat_max = bbox
    k = 4  # Balanced decimation factor

    purl = urllib.parse.urlparse(hdf_url)
    _, file_name = os.path.split(purl.path)

    fs = fsspec.filesystem("http")

    try:
        # Open with a readahead cache to ensure chunk boundaries are fetched in single bursts
        with fs.open(hdf_url, cache_type="readahead", block_size=4 * 1024 * 1024) as fh5:
            with h5py.File(fh5, "r") as hfile:

                if lat_var not in hfile or lon_var not in hfile:
                    raise KeyError(f"L2 file missing {lat_var}/{lon_var}.")
                if var_to_read not in hfile:
                    raise KeyError(f"Variable {var_to_read} not found.")

                # Read arrays out using continuous slicing rather than strided steps over the network
                # This reads the block down completely to prevent chunk-seeking loops
                lats_raw = hfile[lat_var][()]
                lons_raw = hfile[lon_var][()]

                # Decimate in RAM locally (instantaneous)
                lats_small = lats_raw[::k, ::k]
                lons_small = lons_raw[::k, ::k]

                valid_mask_small = (
                        (lats_small >= roi_lat_min) & (lats_small <= roi_lat_max) &
                        (lons_small >= roi_lon_min) & (lons_small <= roi_lon_max)
                )

                if not np.any(valid_mask_small):
                    raise Exception(f'{file_name} does not intersect bbox {bbox}')

                rows_idx, cols_idx = np.where(valid_mask_small)
                rmin, rmax = rows_idx.min() * k, rows_idx.max() * k
                cmin, cmax = cols_idx.min() * k, cols_idx.max() * k

                buf = 20
                lat_shape = lats_raw.shape
                rmin = max(0, rmin - buf)
                rmax = min(lat_shape[0], rmax + buf)
                cmin = max(0, cmin - buf)
                cmax = min(lat_shape[1], cmax + buf)

                if (rmax - rmin < 2) or (cmax - cmin < 2):
                    raise Exception(f'{bbox} yielded empty indices for {file_name}')

                # Slice from local memory arrays
                lats_crop = lats_raw[rmin:rmax, cmin:cmax]
                lons_crop = lons_raw[rmin:rmax, cmin:cmax]

                # Single targeted chunk request over the network for the mask payload data
                var_ds = hfile[var_to_read]
                if len(var_ds.shape) == 3:
                    mask_crop = var_ds[0, rmin:rmax, cmin:cmax]
                else:
                    mask_crop = var_ds[rmin:rmax, cmin:cmax]

                mask_crop = mask_crop.astype(float)
                # Map fill values consistently to NaN
                mask_crop = np.where((mask_crop > 100) | (mask_crop < 0), np.nan, mask_crop)

                # Execute original pyresample layout to preserve geometric accuracy
                swath_def = geometry.SwathDefinition(lons=lons_crop, lats=lats_crop)
                area_def = geometry.AreaDefinition.from_extent(
                    'roi',
                    {'proj': 'latlong', 'datum': 'WGS84'},
                    (50, 50),
                    [roi_lon_min, roi_lat_min, roi_lon_max, roi_lat_max]
                )

                resampled = kd_tree.resample_nearest(
                    swath_def,
                    mask_crop,
                    area_def,
                    radius_of_influence=7000,
                    fill_value=np.nan
                )

                valid_data = resampled[~np.isnan(resampled)]
                if valid_data.size == 0:
                    raise Exception(f'{file_name} contains only nans inside bbox {bbox}')

                cloudy_pixels = valid_data[valid_data == 1].size
                total_valid = valid_data.size

                return int((cloudy_pixels / total_valid) * 100)

    except Exception as e:
        raise


def cloud_coverage(hdf_url: str, bbox: list) -> int:
    # 1. Initialize GDAL environment INSIDE the worker process
    gdal.UseExceptions()
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '300')  # Prevents hanging vsicurl requests

    _, file_name = os.path.split(hdf_url)
    cc = cache.fetch(key=file_name)
    if cc is not None:return cc
    lon_min, lat_min, lon_max, lat_max = bbox
    subdataset_str = f'NETCDF:"/vsicurl/{hdf_url}":CloudMaskBinary'


    # 2. Warp directly from the subdataset string over the network
    # GDAL's C++ engine reads the global polygon, clips the exact byte blocks,
    # and handles the 750m resolution on-the-fly into a memory buffer.
    ds = gdal.Warp(
        '',  # Output to RAM
        subdataset_str,
        format='MEM',
        dstSRS='EPSG:4326', # works here because we are counting pixels not planar metrics
        outputBounds=[lon_min, lat_min, lon_max, lat_max],
        xRes=0.00675, yRes=0.00675,  # 750 meters in decimal degrees
        dstNodata=-128,
        geoloc=True
    )
    if ds is None:
        raise Exception(f'Failed to compute cloud coverage for {hdf_url}')

    data = ds.GetRasterBand(1).ReadAsArray()
    valid_data = data[(data == 0) | (data == 1)]

    if valid_data.size == 0:
        raise Exception(f'Failed to compute cloud coverage for {hdf_url}. No valid data.')
    cc = int((np.count_nonzero(valid_data == 1) / valid_data.size) * 100)
    cache.store(key=file_name, value=cc)
    return cc


def cloud_coverage_batch(urls: list[str], bbox: Iterable[float], max_threads: int = 5, progress: Progress = None):
    results = {}
    master_task = None
    if progress:
        master_task = progress.add_task(
            description=f"[cyan]Computing cloud coverage .... ",
            total=len(urls)
        )
    then = datetime.datetime.now()

    # concurrent.futures is cleaner for CPU-bound resampling
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
        # Map futures to URLs
        future_to_url = {
            executor.submit(cloud_coverage, url, bbox): url
            for url in urls
        }

        for future in concurrent.futures.as_completed(future_to_url, timeout=60):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as e:
                logger.debug(e)
                results[url] = e
            finally:
                if progress and master_task is not None:
                    progress.update(master_task, advance=1)
    now = datetime.datetime.now()
    logger.debug(f'Computed cloud coverage for {len(urls)} granules in {(now-then).total_seconds()} secs')
    if progress and 'master_task' in locals():
        progress.remove_task(master_task)


    return results

