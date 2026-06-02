
from datetime import datetime, timedelta
import numbers
import logging
from rich.progress import Progress
from rapida.components.ntl.variables import generate_variables
from rapida.ntl.fetch import fetch
from scipy.ndimage import uniform_filter
import numpy as np
import rasterio
from scipy.ndimage import label
from rapida.ntl import vis
DELIVERABLES = tuple([g.upper() for g in generate_variables()])
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
    target_data = await fetch(bbox=bbox, nominal_date=nominal_date, progress=progress, deliverable=deliverable,
                                  dst_dir=dst_dir)

    logger.info(f'Fetching baseline imagery for {bbox} {nominal_date}')
    baseline_data = await fetch(bbox=bbox, nominal_date=nominal_date, progress=progress, deliverable='baseline',
                                dst_dir=dst_dir)
    data = {}
    with rasterio.open(target_data) as target, rasterio.open(baseline_data) as base:
        nrt = np.log1p(target.read(1, masked=True))
        baseline = np.log1p(base.read(1, masked=True))
        ssim = pure_numpy_ssim(nrt, baseline, win_size=7)
        anomaly_mask = (ssim < 0.4) & (baseline.filled(0.0) > 1.5) & (~np.ma.getmaskarray(nrt))
        clusters = spatial_filter(anomaly_mask, min_size=5)
        data[target_data] = nrt
        data[baseline_data] = baseline
        data['ssim'] = ssim
        data['anom'] = clusters
        vis.display1(data=data, title=f'All relevant processing levels imagery for {bbox} on {nominal_date.date()}')

