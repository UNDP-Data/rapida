import math
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.enums import ColorInterp
from scipy.ndimage import uniform_filter, gaussian_filter
import logging
from scipy.ndimage import label

logger = logging.getLogger('rapida')



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

def get_intersecting_tiles(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """
    Identifies VIIRS Sinusoidal tiles (h, v) intersecting a geographic bounding box.
    bbox format: (min_lon, min_lat, max_lon, max_lat)
    :return tuple of ints representing pairs of tile coordinates (horizontal, vertical)
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # VIIRS standard sinusoidal grid is approx 10x10 degrees at the equator
    # h runs 0 to 35 (180W to 180E)
    # v runs 0 to 17 (90N to 90S)
    h_min = math.floor((min_lon + 180) / 10)
    h_max = math.floor((max_lon + 180) / 10)
    v_min = math.floor((90 - max_lat) / 10)
    v_max = math.floor((90 - min_lat) / 10)

    tiles = []
    for v in range(max(0, v_min), min(18, v_max + 1)):
        for h in range(max(0, h_min), min(36, h_max + 1)):
            tiles.append(f'h{h:02d}v{v:02d}')

    return tiles

TIMESTAMP_FORMATS = {
    "A1": "%Y%m%d",  # Daily: Year + Julian Day (e.g., 2026134)
    "A2": "%Y%m%d",  # Daily: Year + Julian Day (e.g., 2026134)
    "A3": "%Y%m",  # Monthly: Year + Month (e.g., 202605)
    "A4": "%Y"     # Yearly: Year only (e.g., 2026)
}

def timestamp_format(product_id: str) -> str:
    """Determine the correct temporal format string based on product name."""
    # Check if A1, A2, A3, or A4 is in the product string (e.g., 'VNP46A1')
    for identifier, time_format in TIMESTAMP_FORMATS.items():
        if identifier in product_id:
            return time_format


def write_outage_tif(src_arrays:dict[str, np.array]=None, gt:list = None, dst_path:str=None ) -> bool:
    transform = Affine.from_gdal(*gt)
    label, ar = next(iter(src_arrays.items()))
    height, width = ar.shape
    with rasterio.open(dst_path, mode='w',driver='GTiff',height=height,width=width,
            count=len(src_arrays),
            dtype='float32',
            crs='EPSG:4326',
            transform=transform) as dst:
        dst.update_tags(INTERLEAVE='PIXEL')
        dst.colorinterp = [ColorInterp.undefined] * len(src_arrays)

        for i, e in enumerate(src_arrays.items(), start=1):
            label, array = e
            dst.write(array.astype('float32'), i)
            dst.set_band_description(i, label)



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


