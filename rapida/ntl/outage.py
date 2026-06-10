
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





async def detect_outage_old(
        bbox:tuple[numbers.Number]=None, nominal_date:datetime=None, deliverable:str=None,
        dst_dir:str=None, cmask:bool=True, outage_drop_threshold:int=None, display:bool=False, progress:Progress=None):

    logger.info(f'Fetching best imagery for {deliverable} {bbox}-{nominal_date} ')

    data = {}
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
            baseline_std = extract_bb(image_files=baseline_image_files, sds_name='AllAngle_Composite_Snow_Free_Std', bbox=bbox,
                                            progress=progress)



            data[f'{noaa_const.SDR}_{timestamp}'] = np.log1p(a1)
            data[f'{noaa_const.CM}_{timestamp}'] = cm

            if cmask is True:
                # Create a mask for confident clouds
                is_cloudy = cm > 1
                m |= is_cloudy
            a1_diff, a1z, a1_outage = utils.logdiff_outage(log_monthly_data=a3, log_daily_data=np.log1p(a1), analysis_mask=m,
                                                           percentage_drop=outage_drop_threshold)


            data['a1_diff'] = a1_diff
            data['a1z'] = a1z
            data['a1_outage'] = a1_outage


            if display:
                from rapida.ntl import vis
                vis.display2(data=data, title=f'Outage inputs and results for {deliverable} at {bbox} on {nominal_date.date()}')

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

                    if cmask is True:
                        qf_array = extract_bb(image_files=local_image_files, sds_name='QF_Cloud_Mask', bbox=bbox,
                                              progress=progress).astype('u2')
                        # Cloud mask
                        # Shift right by 6 to bring bits 6 and 7 to the start,
                        # then bitwise AND with 3 (0b11) to isolate just those two bits.
                        # 0=Confident Clear 1=Probably Clear 2=Probably Cloudy 3=Confident Cloudy
                        cloud_confidence = (qf_array >> 6) & 0b11
                        # Create a mask for confident clouds
                        is_cloudy = cloud_confidence == 3
                        m |= is_cloudy




        a1_diff, a1z,  a1_outage = utils.logdiff_outage(log_monthly_data=a3, log_daily_data=a1, analysis_mask=m, percentage_drop=outage_drop_threshold)
        a2_diff, a2z, a2_outage = utils.logdiff_outage(log_monthly_data=a3, log_daily_data=a2, analysis_mask=m, percentage_drop=outage_drop_threshold)

        data['a2_diff'] = a2_diff
        data['a1_diff'] = a1_diff

        data['a2_outage'] = a2_outage
        data['a1_outage'] = a1_outage
        data['a1z'] = a1z

        if display:
            from rapida.ntl import vis
            vis.display2(data=data, title=f'Outage inputs and results for {bbox} on {nominal_date.date()}')

        file_name = utils.get_custom_bbox_label(bbox)
        outage_tif_path = os.path.join(dst_dir, f'{deliverable}_{file_name}.tif')
        write_outage_tif(src_arrays=data, gt=gt,dst_path=outage_tif_path)


async def detect_outage(
        bbox: tuple[numbers.Number] = None, nominal_date: datetime = None, deliverable: str = None,
        dst_dir: str = None, mask_clouds:bool = True, percentage_drop:int = None,
        display: bool = False, progress: Progress = None):

    logger.info(f'Fetching best imagery for {deliverable} {bbox}-{nominal_date} ')

    #fetch daily data, source independent
    # --- 2. FETCH DAILY TARGET DATA ---
    daily_results = await fetch(bbox=bbox, nominal_date=nominal_date, deliverable=deliverable,
                              progress=progress, dst_dir=dst_dir)
    if not daily_results:
        logger.info(f'No imagery was found for {nominal_date:"%Y%m%d"} over {bbox} {deliverable.split("_")[0]}')
        logger.info(f'Consider adjusting source, date or the bounding box')
        return
    for timestamp, product_files in daily_results.items():
        print(timestamp, product_files)

    arrays = {}
    # --- 1. FETCH & PROCESS MONTHLY BASELINE ---
    monthly_nominal_date = nominal_date - timedelta(weeks=6) # the search can handle but its better ot explicit
    monthly_results = await fetch(bbox=bbox, nominal_date=monthly_nominal_date,
                                            progress=progress, deliverable='baseline', dst_dir=dst_dir)

    monthly_timestamp, monthly_files = next(iter(monthly_results.items()))
    monthly_product, monthly_image_files = next(iter(monthly_files.items()))

    monthly_data, gt = extract_bb(image_files=monthly_image_files, sds_name=nasa_const.SUB_DATASETS['A3'],
                                    bbox=bbox, progress=progress, return_gt=True)

    monthly_data_label = f'{monthly_product}_{monthly_timestamp}'
    log_monthly_data = np.log1p(monthly_data)
    arrays[monthly_data_label] = log_monthly_data
    # land water mask 0 = Land & Desert 1 = Land no Desert 2 = Inland Water 3 = Sea Water 5 = Coastal
    # Create a mask for all land pixels including the coastal
    land_water_background = extract_bb(image_files=monthly_image_files, sds_name='Land_Water_Mask', bbox=bbox, progress=progress)
    analysis_mask = (land_water_background == 2) | (land_water_background == 3)



    if 'NOAA' in deliverable:
        target_area = create_area_from_geotransform(gt, monthly_data.shape)
        for timestamp, product_files in daily_results.items():
            daily_data, cloud_mask = read_and_align_sdr_and_cmask(
                sdr_path=product_files[noaa_const.SDR], geo_path=product_files[noaa_const.GEO],
                cmask_path=product_files[noaa_const.CM], target_area=target_area
            )
    elif 'NASA' in deliverable:
        for timestamp, product_files in daily_results.items():
            for product, local_image_files in product_files.items():
                level = product.split('_')[0][-2:] if 'NRT' in deliverable else product[-2:]
                sub_dataset_name = nasa_const.SUB_DATASETS[level]
                daily_data = extract_bb(image_files=local_image_files, bbox=bbox,sds_name=sub_dataset_name,progress=progress)#.astype(np.float32)
                log_daily_data = np.log1p(daily_data)
                daily_data_label = f'{product}_{timestamp}'
                arrays[daily_data_label] = log_daily_data

                if mask_clouds:
                    qf_array = extract_bb(image_files=local_image_files, sds_name='QF_Cloud_Mask',
                                          bbox=bbox, progress=progress).astype('u2')
                    cloud_confidence = (qf_array >> 6) & 0b11
                    analysis_mask |= (cloud_confidence == 3)
                arrays[f'{daily_data_label}_MASK'] = analysis_mask
                log_difference, zscore, outage = utils.logdiff_outage(
                    log_monthly_data=log_monthly_data,log_daily_data=log_daily_data,
                    analysis_mask=analysis_mask,percentage_drop=percentage_drop

                )
                arrays[f'{daily_data_label}_LOGDIFF'] = log_difference
                arrays[f'{daily_data_label}_ZSCORE'] = zscore
                arrays[f'{daily_data_label}_OUTAGE'] = outage



    file_name = utils.get_custom_bbox_label(bbox)
    outage_tif_path = os.path.join(dst_dir, f'{deliverable}_{file_name}.tif')
    write_outage_tif(src_arrays=arrays, gt=gt, dst_path=outage_tif_path)
    # --- 5. UNIFIED DISPLAY & EXPORT ---
    if display:
        from rapida.ntl import vis
        vis.display2(data=arrays,
                     title=f'Outage inputs and results for {deliverable} at {bbox} on {nominal_date.date()}')
