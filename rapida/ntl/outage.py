
from datetime import datetime, timedelta
import numbers
import logging
from rich.progress import Progress
from rapida.ntl.utils import write_outage_tif
from rapida.ntl.fetch import fetch
from rapida.ntl.nasa.io import  extract_bb
from rapida.ntl.nasa import const as nasa_const
import numpy as np
import os
from rapida.ntl.utils import calculate_regional_outage_simplified



logger = logging.getLogger('rapida')





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