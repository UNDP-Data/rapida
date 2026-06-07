import secrets
from calendar import month
from datetime import datetime, timedelta
import numbers
import logging
from rich.progress import Progress
from satpy.resample.base import resample
from rapida.ntl.util import write_outage_tif
from rapida.ntl.fetch import fetch
from rapida.ntl.nasa.io import extract, extract_bb
from rapida.ntl.nasa import const as nasa_const
from scipy.ndimage import uniform_filter, gaussian_filter
import numpy as np
import os
import asyncio
from scipy.ndimage import label
from pathlib import Path



logger = logging.getLogger('rapida')



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


import numpy as np
from scipy.ndimage import gaussian_filter


def calculate_regional_outage_simplified(
        baseline_log_array: np.ndarray,
        nrt_log_array: np.ndarray,
        cloud_mask: np.ndarray,  # Boolean array: True where clouds exist
        sigma: float = 1.25,
        log_drop_threshold: float = -0.69  # approx 50% drop in linear space
) -> np.ndarray:
    """
    Calculates outages using direct log-differencing, which mathematically
    represents proportional radiance loss.
    """
    # 1. Strict Exclusion (No Data is better than Bad Data)
    is_valid = ~np.isnan(nrt_log_array) & ~np.isnan(baseline_log_array) & ~cloud_mask

    # 2. Prevent edge-bleed during smoothing (using your fill method)
    fill_val = np.nanmean(baseline_log_array[is_valid])
    base_filled = np.where(is_valid, baseline_log_array, fill_val)
    nrt_filled = np.where(is_valid, nrt_log_array, fill_val)

    # 3. Apply Gaussian Filter (Handles spatial shifts/jitter)
    base_smooth = gaussian_filter(base_filled, sigma=sigma, truncate=3.0)
    nrt_smooth = gaussian_filter(nrt_filled, sigma=sigma, truncate=3.0)

    # 4. Direct Log Difference (This IS the ratio!)
    # E.g., if NRT is half the Baseline, log1p(NRT) - log1p(Base) is roughly -0.69
    log_diff = nrt_smooth - base_smooth

    # 5. Mask out invalid areas from the final result
    log_diff = np.where(is_valid, log_diff, np.nan)

    # 6. Define the Outage Boolean Map
    # We only care about areas that dropped by more than our threshold
    outage_map = log_diff < log_drop_threshold

    # Optional: You can return log_diff for continuous analysis (severity),
    # or outage_map for binary polygons.
    return log_diff, outage_map

def rigorous_ssim(img1, img2, data_range=12.0) -> np.ndarray:
    """
    Computes standard Wang et al. (2004) SSIM with safety checks for
    MaskedArrays and floating-point catastrophic cancellation.
    """
    # 1. Safely extract masked data and force 64-bit precision
    img1 = img1.filled(0.0) if hasattr(img1, 'filled') else img1
    img2 = img2.filled(0.0) if hasattr(img2, 'filled') else img2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 2. Lock the stability constants globally
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    sigma = 1.5
    trunc = 3.5

    # 3. Compute local means
    mu1 = gaussian_filter(img1, sigma=sigma, truncate=trunc)
    mu2 = gaussian_filter(img2, sigma=sigma, truncate=trunc)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 4. Compute local variances & CLAMP to 0 to prevent float precision collapse
    sigma1_sq = gaussian_filter(img1 ** 2, sigma=sigma, truncate=trunc) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=sigma, truncate=trunc) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma, truncate=trunc) - mu1_mu2

    sigma1_sq = np.maximum(0, sigma1_sq)
    sigma2_sq = np.maximum(0, sigma2_sq)

    # 5. Compute full SSIM map
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den

    return ssim_map

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
                dst_dir:str=None, cmask:bool=True, display:bool=False,  progress:Progress=None):
    data = {}
    logger.info(f'Fetching best imagery for {deliverable} {bbox}-{nominal_date} ')
    baseline_nominal_date = nominal_date - timedelta(weeks=6)
    downloaded_baseline_files = await fetch(bbox=bbox, nominal_date=baseline_nominal_date,
                                            progress=progress, deliverable='baseline', dst_dir=dst_dir)
    baseline_product, baseline_files = next(iter(downloaded_baseline_files.items()))
    timestamp, baseline_image_files = next(iter(baseline_files.items()))
    baseline_array, gt = extract_bb(image_files=baseline_image_files, sds_name=nasa_const.SUB_DATASETS['A3'], bbox=bbox,
                                progress=progress, return_gt=True)


    a3 = np.log1p(baseline_array)

    data[f'{baseline_product}_{timestamp}'] = a3


    downloaded_data_files = await fetch(bbox=bbox, nominal_date=nominal_date, deliverable=deliverable, progress=progress,
                                   dst_dir=dst_dir)


    if 'NOAA' in deliverable: # {timestamp:(product, file_path, size)}
        pass
    else: # NASA, {product:(timestamp:(tiles)}
        for product, results in downloaded_data_files.items():
            level = product[-2:]
            if 'NRT' in deliverable:
                level = product.split('_')[0][-2:]
            for timestamp, local_image_files in results.items():

                ntl_array = extract_bb(image_files=local_image_files,bbox=bbox,sds_name=nasa_const.SUB_DATASETS[level],
                                       progress=progress)
                ntl_array = ntl_array.astype(np.float32)
                if level.lower() == 'a2':
                    a2 = np.log1p(ntl_array)
                    data[f'{product}_{timestamp}'] = a2
                if level.lower() == 'a1':
                    a1 = np.log1p(ntl_array)

                    data[f'{product}_{timestamp}'] = a1
                    qf_array = extract_bb(image_files=local_image_files, sds_name='QF_Cloud_Mask', bbox=bbox,
                                          progress=progress).astype('u2')
                    # land water mask 0 = Land & Desert 1 = Land no Desert 2 = Inland Water 3 = Sea Water 5 = Coastal
                    land_water_bg = (qf_array >> 1) & 0b111
                    # Create a mask for all land pixels
                    m = np.zeros_like(a1).astype(bool)

                    if land_water_bg[land_water_bg==3].size > 0:
                        m = land_water_bg > 1
                    if cmask is True:
                        # Cloud mask
                        # Shift right by 6 to bring bits 6 and 7 to the start,
                        # then bitwise AND with 3 (0b11) to isolate just those two bits.

                        # 0=Confident Clear 1=Probably Clear 2=Probably Cloudy 3=Confident Cloudy
                        cloud_confidence = (qf_array >> 6) & 0b11
                        # Create a mask for confident clouds
                        is_cloudy = cloud_confidence == 3
                        m |= is_cloudy

                    data[f'mask_{timestamp}'] = m

    a1_diff, a1_outage = calculate_regional_outage_simplified(baseline_log_array=a3,nrt_log_array=a1,cloud_mask=m)
    a2_diff, a2_outage = calculate_regional_outage_simplified(baseline_log_array=a3,nrt_log_array=a2,cloud_mask=m)
    data['a1_diff'] = a1_diff
    data['a1_outage'] = a1_outage
    data['a2_diff'] = a2_diff
    data['a2_outage'] = a2_outage
    if display:
        from rapida.ntl import vis
        vis.display1(data=data, title=f'Outage inputs and results for {bbox} on {nominal_date.date()}')
    mword = 'cloud_masked' if cmask else 'landmasked'
    outage_tif_path = os.path.join(dst_dir, f'{deliverable.lower()}_{nominal_date:%Y%m%d}_{mword}.tif')
    write_outage_tif(src_arrays=data, gt=gt,dst_path=outage_tif_path)