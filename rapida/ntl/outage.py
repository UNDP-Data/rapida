
from datetime import datetime, timedelta
import numbers
import logging
from rich.progress import Progress
from rapida.ntl.utils import write_outage_tif
from rapida.ntl.fetch import fetch
from rapida.ntl.nasa.io import  extract_bb
from rapida.ntl.noaa.io import create_area_from_geotransform, read_and_align_sdr_and_cmask
from rapida.ntl.nasa import const as nasa_const
from rapida.ntl.noaa import const as noaa_const
import numpy as np
import os
from rapida.ntl import utils

logger = logging.getLogger('rapida')





async def detect_outage(bbox:tuple[numbers.Number]=None, nominal_date:datetime=None, deliverable:str=None,
                dst_dir:str=None, cmask:bool=True, display:bool=False, outage_drop_threshold:int=None,  progress:Progress=None):
    data = {}
    logger.info(f'Fetching best imagery for {deliverable} {bbox}-{nominal_date} ')
    baseline_nominal_date = nominal_date - timedelta(weeks=6)
    downloaded_baseline_files = await fetch(bbox=bbox, nominal_date=baseline_nominal_date,
                                            progress=progress, deliverable='baseline', dst_dir=dst_dir)
    baseline_product, baseline_files = next(iter(downloaded_baseline_files.items()))
    baseline_timestamp, baseline_image_files = next(iter(baseline_files.items()))
    baseline_array, gt = extract_bb(image_files=baseline_image_files, sds_name=nasa_const.SUB_DATASETS['A3'], bbox=bbox,
                                progress=progress, return_gt=True)

    a3 = np.log1p(baseline_array)

    data[f'{baseline_product}_{baseline_timestamp}'] = a3
    # baseline_std = extract_bb(image_files=baseline_image_files, sds_name='AllAngle_Composite_Snow_Free_Std', bbox=bbox,
    #                                 progress=progress)
    # baseline_num = extract_bb(image_files=baseline_image_files, sds_name='AllAngle_Composite_Snow_Free_Num', bbox=bbox,
    #                           progress=progress)
    lw_bg = extract_bb(image_files=baseline_image_files, sds_name='Land_Water_Mask', bbox=bbox,
                       progress=progress)
    # land water mask 0 = Land & Desert 1 = Land no Desert 2 = Inland Water 3 = Sea Water 5 = Coastal
    # Create a mask for all land pixels including the coastal
    m = (lw_bg == 2) & (lw_bg == 3)

    data['mask'] = m


    downloaded_data_files = await fetch(bbox=bbox, nominal_date=nominal_date, deliverable=deliverable, progress=progress,
                                   dst_dir=dst_dir)
    if not downloaded_data_files:
        return

    if 'NOAA' in deliverable: # {timestamp:{product:file_path}}
        target_area = create_area_from_geotransform(gt, baseline_array.shape)
        for timestamp, entry in downloaded_data_files.items():
            a1, cm = read_and_align_sdr_and_cmask(sdr_path=entry[noaa_const.SDR],geo_path=entry[noaa_const.GEO],
                                         cmask_path=entry[noaa_const.CM], target_area=target_area)

            print(a1.shape, a3.shape, cm.shape)



            data[f'{noaa_const.SDR}_{timestamp}'] = a1
            data[f'{noaa_const.CM}_{timestamp}'] = cm


            # if display:
            #     from rapida.ntl import vis
            #     vis.display2(data=data, title=f'Outage inputs and results for {deliverable} at {bbox} on {nominal_date.date()}')



        return
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
                    # Cloud mask
                    # Shift right by 6 to bring bits 6 and 7 to the start,
                    # then bitwise AND with 3 (0b11) to isolate just those two bits.

                    # 0=Confident Clear 1=Probably Clear 2=Probably Cloudy 3=Confident Cloudy
                    cloud_confidence = (qf_array >> 6) & 0b11
                    #clear =  cloud_confidence == 0
                    #cm = cloud_confidence >= 2
                    if cmask is True:
                        # Create a mask for confident clouds
                        is_cloudy = cloud_confidence == 3
                        m |= is_cloudy




        a1_diff, a1_outage = utils.logdiff_outage(baseline_log_array=a3, nrt_log_array=a1, cloud_mask=m,percentage_drop=outage_drop_threshold)
        a2_diff, a2_outage = utils.logdiff_outage(baseline_log_array=a3, nrt_log_array=a2, cloud_mask=m,percentage_drop=outage_drop_threshold)
        # log_diff,outage_mask, grid_health, confidence = utils.nasa_outage(ntl_log=a1, ntl_baseline_log=a3,
        #                                                                   baseline_std=baseline_std,
        #                                                                   baseline_num=baseline_num,
        #                                                                   no_cloud=clear,
        #                                                                   mask=m)

        data['a2_diff'] = a2_diff
        data['a1_diff'] = a1_diff
        data['a2_outage'] = a2_outage
        data['a1_outage'] = a1_outage

        if display:
            from rapida.ntl import vis
            vis.display2(data=data, title=f'Outage inputs and results for {bbox} on {nominal_date.date()}')

        file_name = utils.get_custom_bbox_label(bbox)
        outage_tif_path = os.path.join(dst_dir, f'{deliverable}_{file_name}.tif')
        write_outage_tif(src_arrays=data, gt=gt,dst_path=outage_tif_path)