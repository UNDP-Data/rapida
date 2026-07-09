
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
from rapida.util.bbox_param_type import get_best_semantic_label
from osgeo import gdal, gdal_array, ogr
from rapida.util.geo import polygonize_raster_mask
from rapida.cli.assess import assess
import click
from rapida.project.project import Project
from tempfile import TemporaryDirectory
import geopandas as gpd
import datetime
logger = logging.getLogger('rapida')
gdal.UseExceptions()


async def detect_outage(
        bbox: tuple[numbers.Number] = None, nominal_date: datetime = None, deliverable: str = None,
        dst_dir: str = None, mask_clouds:bool = True, percentage_drop:int = None, pop_vars:str|tuple[str]=None,
        display: bool = False, progress: Progress = None, year=datetime.datetime.now().year):

    logger.info(f'Fetching best imagery for {deliverable} {bbox}-{nominal_date} ')
    # with open(os.path.join('/tmp', 'bbox.geojson'), "w") as ff:
    #     ff.write(bbox_to_geojson_polygon(*bbox, as_string=True))
    #fetch daily data, source independent
    # --- 2. FETCH DAILY TARGET DATA ---
    daily_results = await fetch(bbox=bbox, nominal_date=nominal_date, deliverable=deliverable,
                              progress=progress, dst_dir=dst_dir)
    if not daily_results:
        logger.info(f'No imagery was found for {nominal_date:"%Y%m%d"} over {bbox} {deliverable.split("_")[0]}')
        logger.info(f'Consider adjusting source, date or the bounding box')
        return


    arrays = {}
    # --- 1. FETCH & PROCESS MONTHLY BASELINE ---
    monthly_nominal_date = nominal_date - timedelta(weeks=6) # the search can handle but its better ot explicit
    monthly_results = await fetch(bbox=bbox, nominal_date=monthly_nominal_date,
                                            progress=progress, deliverable='baseline', dst_dir=dst_dir)

    monthly_timestamp, monthly_files = next(iter(monthly_results.items()))
    monthly_product, monthly_image_files = next(iter(monthly_files.items()))

    monthly_data, gt, rsrs = extract_bb(image_files=monthly_image_files, sds_name=nasa_const.SUB_DATASETS['A3'],
                                    bbox=bbox, progress=progress, return_gt=True)

    monthly_data_label = f'{monthly_product}_{monthly_timestamp}'
    monthly_positive = monthly_data >= 0
    log_monthly_data = np.zeros_like(monthly_data)
    log_monthly_data[monthly_positive] = np.log1p(monthly_data[monthly_positive])
    arrays[monthly_data_label] = log_monthly_data
    # land water mask 0 = Land & Desert 1 = Land no Desert 2 = Inland Water 3 = Sea Water 5 = Coastal
    # Create a mask for all land pixels including the coastal
    land_water_background = extract_bb(image_files=monthly_image_files, sds_name='Land_Water_Mask', bbox=bbox, progress=progress)
    analysis_mask = (land_water_background == 2) | (land_water_background == 3)
    analysis_mask |= ~monthly_positive

    if 'NOAA' in deliverable:
        target_area = create_area_from_geotransform(gt, monthly_data.shape)

        # 1. Initialize master COMPOSITE arrays
        # Use NaNs so unobserved pixels remain NaN
        log_diff = np.full_like(monthly_data, np.nan)
        zscore = np.full_like(monthly_data, np.nan)
        outage = np.zeros_like(monthly_data, dtype=bool)

        # 2. Master tracking mask (True if a pixel gets AT LEAST ONE clear observation)
        valid_mask = np.zeros_like(analysis_mask, dtype=bool)

        for timestamp, product_files in daily_results.items():


            daily_data, cloud_mask = read_and_align_sdr_and_cmask(
                sdr_path=product_files[noaa_const.SDR], geo_path=product_files[noaa_const.GEO],
                cmask_path=product_files[noaa_const.CM], target_area=target_area
            )


            # 3. Create a STRICTLY DAILY invalid mask
            # It inherits the base analysis_mask (water/bg), but does NOT inherit yesterday's clouds
            daily_invalid_mask = np.isnan(daily_data) | analysis_mask
            daily_data_label = f'{noaa_const.SDR}_{timestamp}'
            if mask_clouds is True:
                is_cloudy = cloud_mask > 1
                cloudy_percentage = is_cloudy[is_cloudy].size/is_cloudy.size *100
                if round(cloudy_percentage ) >= 95:
                    logger.info(f'There are too many clouds in the bbox: {cloudy_percentage:.2f}%. Skipping cloud mask' )
                else:
                    daily_invalid_mask |= is_cloudy
            daily_invalid_percentage = daily_invalid_mask[daily_invalid_mask].size/daily_invalid_mask.size*100
            if round(daily_invalid_percentage) >= 97:
                logger.info(f'There are too many masked/invalid pixels in the bbox for {daily_data_label}: {daily_invalid_percentage}%. It will be skipped' )
                continue

            log_daily_data = np.zeros_like(daily_data)
            log_daily_data[~daily_invalid_mask] = np.log1p(daily_data[~daily_invalid_mask])

            # Save individual daily raw data if needed

            arrays[daily_data_label] = log_daily_data

            # Run the NTL statistics engine on today's clear pixels
            granule_logdiff, granule_zscore, granule_outage = utils.logdiff_outage(
                log_monthly_data=log_monthly_data, log_daily_data=log_daily_data,
                analysis_mask=daily_invalid_mask, percentage_drop=percentage_drop
            )

            # 4. Update the Master Composites
            # We define what was valid TODAY
            daily_valid = ~daily_invalid_mask

            # Isolate pixels that are valid in THIS granule AND haven't been mapped yet today
            is_first_time = daily_valid & ~valid_mask

            # Lock in the data ONLY for these first-time pixels
            log_diff[is_first_time] = granule_logdiff[is_first_time]
            zscore[is_first_time] = granule_zscore[is_first_time]
            outage[is_first_time] = granule_outage[is_first_time]

            # Update our tracking mask to protect these pixels from future overlaps
            valid_mask |= daily_valid



        ts = f'{nominal_date:%Y%m%d}'

        # 5. Save the final merged dataset outside the loop
        # Invert the valid mask so True = "We never saw this pixel clearly"
        arrays[f'{ts}_MASK'] = ~valid_mask
        arrays[f'{ts}_LOGDIFF'] = log_diff
        arrays[f'{ts}_ZSCORE'] = zscore
        arrays[f'{ts}_OUTAGE'] = outage

    elif 'NASA' in deliverable:
        for timestamp, product_files in daily_results.items():
            for product, local_image_files in product_files.items():
                level = product.split('_')[0][-2:] if 'NRT' in deliverable else product[-2:]
                sub_dataset_name = nasa_const.SUB_DATASETS[level]
                daily_data = extract_bb(image_files=local_image_files, bbox=bbox,sds_name=sub_dataset_name,progress=progress)
                positive = daily_data>=0
                log_daily_data = np.zeros_like(daily_data)
                log_daily_data[positive] = np.log1p(daily_data[positive])
                log_daily_data = np.log1p(daily_data)

                daily_data_label = f'{product}_{timestamp}'
                arrays[daily_data_label] = log_daily_data

                if mask_clouds:
                    qf_array = extract_bb(image_files=local_image_files, sds_name='QF_Cloud_Mask',
                                          bbox=bbox, progress=progress).astype('u2')
                    cloud_confidence = (qf_array >> 6) & 0b11
                    is_cloudy = cloud_confidence == 3
                    arrays['CLOUD_MASK'] = is_cloudy
                    analysis_mask |= is_cloudy
                arrays[f'{daily_data_label}_MASK'] = analysis_mask
                log_difference, zscore, outage = utils.logdiff_outage(
                    log_monthly_data=log_monthly_data,log_daily_data=log_daily_data,
                    analysis_mask=analysis_mask,percentage_drop=percentage_drop

                )
                arrays[f'{daily_data_label}_LOGDIFF'] = log_difference
                arrays[f'{daily_data_label}_ZSCORE'] = zscore
                arrays[f'{daily_data_label}_OUTAGE'] = outage

    # --- 5. UNIFIED DISPLAY & EXPORT ---
    #file_name = utils.get_custom_bbox_label(bbox)
    file_name = get_best_semantic_label(bbox=bbox)
    outage_tif_path = os.path.join(dst_dir, f'{deliverable}_{file_name}.tif')
    outage_gpkg_path = os.path.join(dst_dir, f'{deliverable}_{file_name}.gpkg')
    write_outage_tif(src_arrays=arrays, gt=gt, dst_path=outage_tif_path)

    if pop_vars:
        # vectorize outage
        gtiff_driver = gdal.GetDriverByName("GTiff")

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(outage_gpkg_path)), exist_ok=True)

        if not os.path.exists(outage_gpkg_path):
            # 1. Ask GDAL to sniff out the right driver name based on the path structure/extension
            v_driver = ogr.GetDriverByName('GPKG')

            # 2. Use the driver to initialize the empty file header
            with v_driver.CreateDataSource(outage_gpkg_path) as init_ds:
                pass  # Successfully generated empty stub container
        for k, arr in arrays.items():
            if 'outage' in k.lower():
                # Ensure data type is compatible (e.g., forcing int64/bool to int32/uint8 if needed)
                if arr.dtype == bool:
                    arr = arr.astype("uint8")
                elif arr.dtype == "int64":
                    arr = arr.astype("int32")  # GDAL standard doesn't always love 64-bit ints
                gdal_dt = gdal_array.NumericTypeCodeToGDALTypeCode(arr.dtype)
                rows, cols = arr.shape
                # Generate a unique virtual memory path
                vsimem_path = f"/vsimem/{k}.tif"


                try:
                    # Create, write, and safely close the dataset within the context manager
                    with gtiff_driver.Create(vsimem_path, cols, rows, 1, gdal_dt) as ds:
                        ds.SetGeoTransform(gt)
                        ds.SetSpatialRef(rsrs)
                        band = ds.GetRasterBand(1)
                        band.WriteArray(arr)
                        band.FlushCache()

                    # The dataset is now closed, but the file exists in /vsimem/
                    polygonize_raster_mask(
                        raster_ds=vsimem_path,  # Passes the string path
                        dst_dataset=outage_gpkg_path,
                        dst_layer=k,
                        geom_type=ogr.wkbPolygon
                    )

                    with TemporaryDirectory(dir=dst_dir, delete=True) as project_folder:
                        project = Project(path=project_folder, polygons=outage_gpkg_path,
                                          comment='temp project for outage zonal stats')
                        with click.Context(assess) as ctx:
                            ctx.ensure_object(dict)
                            ctx.obj['progress'] = progress
                            # 2. Use invoke. Do NOT pass 'ctx' manually here.
                            # Click intercepts this and injects it as the first argument automatically.
                            ctx.invoke(
                                assess,
                                components=('population',),
                                variables=pop_vars,
                                year=year,
                                project=project.path,
                                force=False
                            )
                            stat_gpkg_path = os.path.join(project_folder, 'data', f'{project.name}.gpkg')
                            pop_stat_gdf = gpd.read_file(stat_gpkg_path, layer='stats.population')

                            pop_stat_gdf = pop_stat_gdf.to_crs('EPSG:4326')

                            pop_stat_gdf.to_file(
                                filename=outage_gpkg_path,
                                driver="gpkg",
                                engine="pyogrio",
                                mode="w",
                                layer=k,
                                promote_to_multi=True,
                                index=False
                            )


                finally:
                    # Unlink (delete) the virtual file to free up system memory
                    gdal.Unlink(vsimem_path)
    if display:
        from rapida.ntl import vis
        vis.display2(data=arrays,
                     title=f'Outage inputs and results for {deliverable} at {bbox} on {nominal_date.date()}')
