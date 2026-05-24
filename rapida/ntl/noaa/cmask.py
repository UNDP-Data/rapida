from pyresample import geometry, kd_tree
import os.path
import urllib.parse
import h5py
import logging
import fsspec
from shapely import wkt, box
from typing import Iterable, Any
import concurrent
import numpy as np
from rich.progress import Progress

logger = logging.getLogger(__name__)


def bbox_in_hdf(hdf_url: str, bbox: Iterable[float]):
    fs = fsspec.filesystem("http")
    purl = urllib.parse.urlparse(hdf_url)
    _, filename = os.path.split(purl.path)

    with fs.open(hdf_url, block_size=1024 * 1024) as f:
        with h5py.File(f, "r") as hfile:
            # Now it reads at HTTP speeds without the boto3 overhead
            bounds_poly = wkt.loads(hfile.attrs['geospatial_bounds'].decode('utf-8'))
            bbox_poly = box(*bbox, ccw=True)
            if not bbox_poly.within(bounds_poly):
                return False
            return True

def cloud_coverage(hdf_url: str, bbox: Iterable[float],
                   lon_var: str = 'Longitude', lat_var: str = 'Latitude',
                   var_to_read: str = 'CloudMaskBinary'):
    """
    Computes cloud coverage percentage for a given BBox.
    Note: Removed 'progress' from here; sub-tasks shouldn't manage the master bar.
    """
    roi_lon_min, roi_lat_min, roi_lon_max, roi_lat_max = bbox
    k = 5  # Decimation factor
    purl = urllib.parse.urlparse(hdf_url)
    rel_path, file_name = os.path.split(purl.path)
    fs = fsspec.filesystem("http")
    try:
        with fs.open(hdf_url, block_size=1024 * 1024) as fh5:
            with h5py.File(fh5, "r") as hfile:

                # 1. Verification Step: Ensure coordinates actually exist in this file
                if lat_var not in hfile or lon_var not in hfile:
                    raise KeyError(f"L2 file missing {lat_var}/{lon_var}. Must use GDNBO file for coords.")
                if var_to_read not in hfile:
                    raise KeyError(f"Variable {var_to_read} not found in this file.")

                # 2. Fast Index Search (Decimated)
                lats_small = hfile[lat_var][::k, ::k]
                lons_small = hfile[lon_var][::k, ::k]

                valid_mask_small = (
                        (lats_small >= roi_lat_min) & (lats_small <= roi_lat_max) &
                        (lons_small >= roi_lon_min) & (lons_small <= roi_lon_max)
                )

                if not np.any(valid_mask_small):
                    raise Exception(f'{file_name} does not intersect bbox {bbox}')

                # 3. Calculate bounding indices with decimation factor scaling
                rows_idx, cols_idx = np.where(valid_mask_small)
                rmin, rmax = rows_idx.min() * k, rows_idx.max() * k
                cmin, cmax = cols_idx.min() * k, cols_idx.max() * k

                # 4. Apply Buffer and Clamp to Array Shape
                buf = 15
                lat_shape = hfile[lat_var].shape

                rmin = max(0, rmin - buf)
                rmax = min(lat_shape[0], rmax + buf)
                cmin = max(0, cmin - buf)
                cmax = min(lat_shape[1], cmax + buf)

                # DEFENSE 1: Check for degenerate (empty or 1D) slices
                if (rmax - rmin < 2) or (cmax - cmin < 2):
                    raise Exception(f'{bbox} has yielded empty indices for {file_name} ')

                # 5. Extract Crops
                lats_crop = hfile[lat_var][rmin:rmax, cmin:cmax]
                lons_crop = hfile[lon_var][rmin:rmax, cmin:cmax]

                # DEFENSE 2: Handle potential 3D variable shapes
                var_shape = hfile[var_to_read].shape
                if len(var_shape) == 3:
                    mask_crop = hfile[var_to_read][0, rmin:rmax, cmin:cmax]
                else:
                    mask_crop = hfile[var_to_read][rmin:rmax, cmin:cmax]

                # DEFENSE 3: Mask out fill values to prevent pyresample skewing
                mask_crop = mask_crop.astype(float)
                mask_crop = np.where(mask_crop > 100, np.nan, mask_crop)

                # 6. Pyresample Execution
                swath_def = geometry.SwathDefinition(lons=lons_crop, lats=lats_crop)

                # Setup target grid (50x50 pixels is standard for quick ROI stats)
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

                # 7. Final Coverage Computation
                valid_data = resampled[~np.isnan(resampled)]
                if valid_data.size == 0:
                    raise Exception(f'{file_name} contains only nans inside bbox {bbox}')

                # Calculate percentage. Ensure '1' actually means cloudy in your specific L2 spec!
                cloudy_pixels = valid_data[valid_data == 1].size
                total_valid = valid_data.size

                return int((cloudy_pixels / total_valid) * 100)

    except Exception as e:
        raise


def cloud_coverage_batch(urls: list[str], bbox: Iterable[float], max_threads: int = 5, progress: Progress = None):
    results = {}
    master_task = None
    if progress:
        master_task = progress.add_task(
            description=f"[cyan]Computing cloud coverage .... ",
            total=len(urls)
        )

    # concurrent.futures is cleaner for CPU-bound resampling
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Map futures to URLs
        future_to_url = {
            executor.submit(cloud_coverage, url, bbox): url
            for url in urls
        }

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as e:
                logger.debug(e)
                results[url] = e
            finally:
                if progress and master_task is not None:
                    progress.update(master_task, advance=1)

    return results
