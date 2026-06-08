import math
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.enums import ColorInterp
from scipy.ndimage import uniform_filter, gaussian_filter
import logging
from scipy.ndimage import label
from scipy.ndimage import convolve
from rapida.util.http_get_json import http_get_json
logger = logging.getLogger('rapida')


def get_custom_bbox_label(bbox: tuple[float, float, float, float]) -> str:
    minlon, minlat, maxlon, maxlat = bbox

    lon = (minlon + maxlon) * .5
    lat = (minlat + maxlat) * .5

    # 1. Calculate the longest span of the bbox in degrees
    max_span = max(maxlon - minlon, maxlat - minlat)

    # 2. Map the size to a Nominatim zoom level (Adjusted for real-world sizes)
    if max_span > 15.0:
        zoom = 3  # Country level (Massive areas like the whole USA or Europe)
    elif max_span > 1.0:
        zoom = 5  # State/Region level (Catches Puerto Rico's 2.6 span perfectly)
    elif max_span > 0.2:
        zoom = 8  # County/District level (Large metro areas)
    else:
        zoom = 12  # City/Town/Village level (Small localized bboxes)

    # 3. Add the zoom parameter to the URL
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&zoom={zoom}&format=json"

    headers = {'User-Agent': 'UNDP/RAPIDA'}

    try:
        data = http_get_json(url=url, timeout=30, headers=headers)
        address = data.get('address', {})

        country_iso2 = address.get('country_code', 'XX').upper()
        admin1 = address.get('state', address.get('state_district'))
        admin2 = address.get('county', address.get('district'))
        admin3 = address.get('city', address.get('town', address.get('village')))

        # 4. Clean up the string formatting to avoid "US_-"
        names = []
        for e in [admin1, admin2, admin3]:
            if e not in ['', None]:
                names.append(e)

        # Join the admins with dashes, then attach the country code
        admin_str = "-".join(names)
        if admin_str:
            return f"{country_iso2}_{admin_str}"
        else:
            return country_iso2

    except Exception as e:
        return f"Error_{e}"


def disk_filter(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Applies a uniform circular (pillbox) blur.
    A radius of 1 creates a 3x3 cross. A radius of 2 creates a 5x5 circle.
    """
    # 1. Generate a circular mask using an orthogonal grid
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2

    # 2. Create the kernel and normalize it so it sums to 1.0
    kernel = mask.astype(np.float64)
    kernel /= kernel.sum()

    # 3. Convolve the image with the circular kernel
    return convolve(image, kernel, mode='reflect')


def noaa_outage(
        nrt_array: np.ndarray,
        baseline_array: np.ndarray,
        baseline_std: np.ndarray,
        cmask: np.ndarray,
        lwm: np.ndarray,
        baseline_is_log: bool = True,
        relative_noise_floor: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detects outages using pixel-specific Z-scores and global atmospheric shift.

    Args:
        nrt_array: Near Real-Time VIIRS array (linear scale).
        baseline_array: Level-3 baseline array.
        baseline_std: Level-3 baseline standard deviation array.
        cmask: Resampled Cloud Mask (0=Clear, 1=Probably Clear).
        lwm: Resampled Land Water Mask (0, 1, 5 = Land).
        baseline_is_log: True if baseline_array is already np.log1p().
        relative_noise_floor: Minimum standard deviation limit.

    Returns:
        final_outages (np.ndarray): Boolean mask of confirmed outages.
        z_map_display (np.ndarray): Continuous array of Z-scores for visualization.
        grid_health (np.ndarray): Categorical array (0=Healthy, 1=Dimming, 2=Outage).
    """
    # 1. Align Data Spaces
    a3_log = baseline_array if baseline_is_log else np.log1p(baseline_array)
    nrt_log = np.log1p(nrt_array)

    # 2. Master Validity Mask
    is_land = np.isin(lwm, [0, 1, 5])
    is_clear = (cmask <= 1)
    has_signal = a3_log > np.log1p(0.5)

    master_mask = is_land & is_clear & has_signal & ~np.isnan(a3_log) & ~np.isnan(nrt_log)

    if not np.any(master_mask):
        logger.warning("No valid clear land pixels found in the target area.")
        return np.zeros_like(nrt_array, dtype=bool), np.full_like(nrt_array, np.nan), np.zeros_like(nrt_array,
                                                                                                    dtype=np.uint8)

    # 3. Atmospheric Normalization
    log_diff = nrt_log - a3_log
    atmos_shift = np.nanmedian(log_diff[master_mask])

    log_pred = a3_log + atmos_shift
    log_res = nrt_log - log_pred

    # 4. Pixel-Specific Z-Score
    a3_linear = np.expm1(a3_log)
    log_std = np.maximum(baseline_std / (a3_linear + 1.0), relative_noise_floor)
    z_map_log = log_res / log_std

    # 5. Multi-Tiered Classification Logic
    is_dimming = (log_res < -0.3) | (z_map_log < -2.0)
    is_outage = (log_res < -0.7) | (z_map_log < -3.0)

    # 6. Apply the Mask to Build the Heatmap
    grid_health = np.zeros_like(log_res, dtype=np.uint8)
    grid_health[is_dimming & master_mask] = 1
    grid_health[is_outage & master_mask] = 2

    # 7. Spatial Cleanup
    final_outages = spatial_filter((grid_health == 2), min_size=2)

    # 8. Clean visualization map
    z_map_display = np.where(master_mask, z_map_log, np.nan)

    return final_outages, z_map_display, grid_health


def nasa_outage_old(
        ntl_log: np.ndarray,
        ntl_baseline_log: np.ndarray,
        mask: np.ndarray,  # Your 'm' mask: True = Cloud/Water (Ignore)
        dimming_threshold: float = -0.69, # ~ 25 percent
        outage_threshold: float = -1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates outages using spatial smoothing and direct log-differencing.

    Args:
        ntl_log: Near Real-Time VIIRS array (linear scale).
        ntl_baseline_log: Level-3 baseline array.
        mask: Boolean array where True means the pixel should be ignored.
        baseline_is_log: True if baseline_array is already np.log1p().
        sigma: Gaussian blur radius to handle registration jitter.
        dimming_threshold: Log-ratio threshold for dimming events. default e**-0.3= 0.74; then 1 - .74 = .26 26% loss of light
        outage_threshold: Log-ratio threshold for severe outages. default = e**-0.69 = -.5 ; 1-.5 = .5 50% loss of light

    Returns:
        final_outages (np.ndarray): Boolean mask of confirmed outages.
        diff_display (np.ndarray): Continuous array of log-differences for visualization.
        grid_health (np.ndarray): Categorical array (0=Healthy, 1=Dimming, 2=Outage).
    """
    # 1. Align Data Spaces
    a3_log = ntl_baseline_log
    nrt_log = ntl_log

    # 2. Master Validity Mask
    # We invert your invalid_mask (~invalid_mask) so True = Good Pixel
    has_signal = a3_log > np.log1p(0.5)
    is_valid = ~mask & has_signal & ~np.isnan(a3_log) & ~np.isnan(nrt_log)

    if not np.any(is_valid):
        logger.warning("No valid pixels found after applying the invalid mask.")
        return np.zeros_like(ntl_log, dtype=bool), np.full_like(ntl_log, np.nan), np.zeros_like(ntl_log, dtype=np.uint8)

    # 3. Prevent edge-bleed during smoothing
    fill_val = np.nanmean(a3_log[is_valid])
    base_filled = np.where(is_valid, a3_log, fill_val)
    nrt_filled = np.where(is_valid, nrt_log, fill_val)

    # 4. Apply Gaussian Filter (Handles spatial shifts/jitter)
    # base_smooth = gaussian_filter(base_filled, sigma=sigma, truncate=3.0)
    # nrt_smooth = gaussian_filter(nrt_filled, sigma=sigma, truncate=3.0)

    # 4. Apply Disk Filter (The "Smooth but not Squary" method)
    # radius=1 gives a tight circle (5 pixels). radius=2 gives a wider circle (13 pixels).
    base_smooth = disk_filter(base_filled, radius=1)
    nrt_smooth = disk_filter(nrt_filled, radius=1)

    # 5. Direct Log Difference
    log_diff = nrt_smooth - base_smooth

    is_dimming = (log_diff < dimming_threshold)
    is_outage = (log_diff < outage_threshold)

    # 7. Apply the Mask to Build the Heatmap
    grid_health = np.zeros_like(log_diff, dtype=np.uint8)
    grid_health[is_dimming & is_valid] = 1
    grid_health[is_outage & is_valid] = 2

    # Since Gaussian blur inherently mitigates spatial jitter,
    # we pull the boolean mask directly from the grid_health array.
    final_outages = (grid_health == 2)

    # 8. Clean visualization map
    diff_display = np.where(is_valid, log_diff, np.nan)

    return diff_display, final_outages,  grid_health





def nasa_outage(
        ntl_log: np.ndarray,
        ntl_baseline_log: np.ndarray,
        baseline_std: np.ndarray,  # <-- NEW: A3 _Std band required here
        baseline_num: np.ndarray,  # <-- NEW: A3 _Num band required here
        no_cloud:np.ndarray,
        mask: np.ndarray,
        dimm_perc_drop = 50,
        outage_perc_drop = 80
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates outages using spatial smoothing and variance-scaled thresholding.
    """
    a3_log = ntl_baseline_log
    nrt_log = ntl_log

    # 1. Gate
    #is_valid = ~mask & ~np.isnan(a3_log) & ~np.isnan(nrt_log)# & (a3_log>0)
    is_valid = ~np.isnan(nrt_log) & ~np.isnan(a3_log) & ~mask
    if not np.any(is_valid):
        logger.warning("No valid pixels found.")
        return np.full_like(ntl_log, np.nan), np.zeros_like(ntl_log, dtype=bool), np.zeros_like(ntl_log,
                                                                                                dtype=np.uint8), np.zeros_like(
            ntl_log, dtype=np.float32)

    # 2. Prevent edge-bleed during smoothing
    fill_val = np.nanmean(a3_log[is_valid])
    base_filled = np.where(is_valid, a3_log, fill_val)
    nrt_filled = np.where(is_valid, nrt_log, fill_val)

    # 3. Spatial Smoothing
    base_smooth = disk_filter(base_filled, radius=1)
    nrt_smooth = disk_filter(nrt_filled, radius=1)

    # 4. Direct Log Difference
    log_diff = np.zeros_like(ntl_log, dtype=np.float32)
    log_diff[is_valid] = nrt_smooth[is_valid] - base_smooth[is_valid]


    log_outage_threshold = np.log(1.0 - (outage_perc_drop / 100.0))
    log_dimm_threshold = np.log(1.0 - (dimm_perc_drop / 100.0))

    # 6. Apply Dynamic Classification
    # We are now comparing the log_diff array against the dynamic threshold arrays
    grid_health = np.zeros_like(ntl_log, dtype=np.uint8)
    grid_health[is_valid & (log_diff < log_dimm_threshold)] = 1
    grid_health[is_valid & (log_diff < log_outage_threshold)] = 2

    final_outages = (grid_health == 2)

    # --- PHYSICAL CONFIDENCE SCORE ---

    # 7. Compute Confidence adapted for dynamic thresholds
    confidence = np.zeros_like(log_diff, dtype=np.float32)

    outage_pixels = (grid_health == 2) & is_valid

    # 1. Evidence: Signal-to-Noise Ratio (Is the drop larger than local noise?)
    # We use a 3-sigma rule (SNR=3.0 is 100% certainty)
    std_safe = np.where(baseline_std < 0.05, 0.05, baseline_std)  # Floor to avoid div by zero
    snr = np.abs(log_diff[outage_pixels]) / std_safe[outage_pixels]
    conf_snr = np.clip(snr / 3.0, 0.0, 1.0)

    # 2. Evidence: Baseline Robustness (Do we have enough history?)
    # Assume 'a3_num' is the count of clear days from your VNP46A3 product.
    # 15 days is high confidence, 1 day is low.
    conf_history = np.clip(baseline_num[outage_pixels] / 15.0, 0.0, 1.0)

    # 3. Evidence: Observability (Was the NTL pixel actually observed clearly?)
    # 1.0 if Clear, 0.0 if Cloud-Contaminated/Shadow
    conf_obs = np.where(no_cloud[outage_pixels] == 0, 1.0, 0.0)

    # FINAL CONFIDENCE: The product of all evidences
    # If any evidence is weak, the total confidence drops.
    confidence[outage_pixels] = conf_snr * conf_history * conf_obs

    # 8. Clean visualization map
    diff_display = np.where(is_valid, log_diff, np.nan)

    return diff_display, final_outages, grid_health, confidence

def logdiff_outage(
        baseline_log_array: np.ndarray,
        nrt_log_array: np.ndarray,
        cloud_mask: np.ndarray,  # Boolean array: True where clouds exist
        sigma: float = 1.25,
        percentage_drop: float = 50
) -> np.ndarray:
    """
    Calculates outages using direct log-differencing, which mathematically
    represents proportional radiance loss.
    """
    remaining_fraction = 1.0 - (percentage_drop / 100.0)
    log_drop_threshold = np.log(remaining_fraction)

    # 1. Strict Exclusion (No Data is better than Bad Data)
    is_valid = ~np.isnan(nrt_log_array) & ~np.isnan(baseline_log_array) & ~cloud_mask

    # 2. Prevent edge-bleed during smoothing (using your fill method)
    fill_val = np.nanmean(baseline_log_array[is_valid])
    base_filled = np.where(is_valid, baseline_log_array, fill_val)
    nrt_filled = np.where(is_valid, nrt_log_array, fill_val)

    # 3. Apply Gaussian Filter (Handles spatial shifts/jitter)
    # base_smooth = gaussian_filter(base_filled, sigma=sigma, truncate=3.0)
    # nrt_smooth = gaussian_filter(nrt_filled, sigma=sigma, truncate=3.0)

    # 3. Spatial Smoothing
    base_smooth = disk_filter(base_filled, radius=1)
    nrt_smooth = disk_filter(nrt_filled, radius=1)

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


