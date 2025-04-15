import pyogrio.core
import multiprocessing
from cbsurge import constants
from osgeo import gdal, ogr, osr
from osgeo_utils.gdal_calc import Calc
import rasterio
from cbsurge.util.proj_are_equal import proj_are_equal
import os
import tempfile
from rich.progress import Progress
import logging
logger = logging.getLogger('rapida')
gdal.UseExceptions()


def isgdal(src=None, file_type=None):
    """
    Test if src is a GDAL raster
    :param src:
    :return:
    """
    file_types = dict(raster=gdal.OF_RASTER, vector=gdal.OF_VECTOR)
    try:
        ftype = file_types[file_type]
        try:
            with gdal.OpenEx(src, ftype | gdal.OF_READONLY):
                return True
        except RuntimeError as re:
            return False
    except KeyError:
        logger.error(f'Wrong file_type arg. Valid values are {",".join(list(file_types))}')
        raise

is_raster = lambda src: isgdal(src=src, file_type='raster')
is_vector = lambda src: isgdal(src=src, file_type='vector')


def gdal_callback(complete, message, data):
    if data:
        progressbar, task, timeout_event = data
        if progressbar is not None and task is not None:
            progressbar.update(task, completed=int(complete * 100))
        if timeout_event and timeout_event.is_set():
            logger.info(f'GDAL was signalled to stop...')
            return 0

def import_raster(source=None, dst=None, target_srs=None,
                  x_res: int = constants.DEFAULT_MASK_RESOLUTION_METERS,
                  y_res: int = constants.DEFAULT_MASK_RESOLUTION_METERS,
                  crop_ds=None, crop_layer_name=None, return_handle=False,
                  progress=None,
                  **kwargs
                  ):
    """
    Import a raster into RAPIDA project as a GeoTiff
    :param target_srs:
    :param source:
    :param dst:
    :param x_res:
    :param y_res:
    :param crop_ds:
    :param crop_layer_name:
    :param return_handle:
    :param kwargs:
    :return:
    """

    assert crop_layer_name not in ('', None), f'crop_layer_name: {crop_layer_name} is invalid'
    assert is_vector(crop_ds), f'{crop_ds} is not a vector dataset'
    tout = multiprocessing.Event()
    rtask = None
    if progress is not None:
        rtask = progress.add_task(f'Reprojecting {source}')
    try:
        should_crop = crop_ds is not None and crop_layer_name not in ('', None)
        if should_crop:
            info = pyogrio.core.read_info(crop_ds, layer=crop_layer_name)
            out_bounds = list(map(float, info['total_bounds']))
            out_bounds_srs = osr.SpatialReference()
            out_bounds_srs.ImportFromWkt(info['crs'])

        else:
            out_bounds = None
            out_bounds_srs = None



        warp_options = dict(
            format='GTiff',
            xRes=x_res,
            yRes=y_res,
            creationOptions=constants.GTIFF_CREATION_OPTIONS.update({'COMPRESS':'NONE'}),
            cutlineDSName=crop_ds,
            cutlineLayer=crop_layer_name,
            cropToCutline=should_crop,
            targetAlignedPixels=True,
            outputBounds=out_bounds,
            outputBoundsSRS=out_bounds_srs,
            warpOptions={'OPTIMIZE_SIZE':'YES'},
            errorThreshold=2,


        )
        warp_options.update(kwargs)
        if progress is not None and rtask is not None:
            callback_dict = dict(
                callback=gdal_callback,
                callback_data = (progress, rtask, tout)
            )
            warp_options.update(callback_dict)


        with gdal.OpenEx(source, gdal.OF_RASTER | gdal.OF_READONLY) as src_ds:
            src_srs = src_ds.GetSpatialRef()
            prj_equal = proj_are_equal(src_srs, target_srs)
            if not prj_equal:
                # reproject raster mask to target projection
                logger.debug(f'Reprojecting {source} to {target_srs.GetAuthorityName(None)}:{target_srs.GetAuthorityCode(None)}')
                warp_options.update(dict(dstSRS=target_srs))
                rds = gdal.Warp(destNameOrDestDS=dst, srcDSOrSrcDSTab=src_ds, **warp_options)
            else:
                rds = gdal.Warp(destNameOrDestDS=dst, srcDSOrSrcDSTab=src_ds, **warp_options)

            assert os.path.exists(dst), f'Failed to align {source}'
            if return_handle:
                return rds
            else:
                rds = None
    except KeyboardInterrupt:
        tout.set()
        raise
    except (RuntimeError, Exception) as re:
        if 'user terminated' in str(re).lower():
            logger.info(f'Reprojection was aborted. Cleaning up')
            if os.path.exists(dst):
                os.remove(dst)
                os.remove(source)
                raise KeyboardInterrupt
    finally:
        if progress is not None and rtask is not None:
            progress.remove_task(rtask)

def polygonize_raster_mask(raster_ds=None, band=1, dst_dataset=None, dst_layer=None, geom_type=ogr.wkbMultiPolygon):

    with gdal.OpenEx(raster_ds, gdal.OF_RASTER|gdal.OF_READONLY) as rds:
        with gdal.OpenEx(dst_dataset, gdal.OF_VECTOR | gdal.OF_UPDATE) as vds:
            logger.info(f'Polygonizing {raster_ds} ')
            mband = rds.GetRasterBand(band)
            mask_lyr = vds.CreateLayer(dst_layer, geom_type=geom_type, srs=rds.GetSpatialRef())

            r = gdal.Polygonize(srcBand=mband, maskBand=mband,outLayer=mask_lyr,iPixValField=-1)

            assert r == 0, f'Failed to polygonize {raster_ds}'
            for feature in mask_lyr:
                geom = feature.GetGeometryRef()
                simplified_geom = geom.Simplify(constants.DEFAULT_MASK_POLYGONIZATION_SMOOTHING_BUFFER)  # Use SimplifyPreserveTopology(tolerance) if needed
                smoothed_geom = simplified_geom.Buffer(constants.DEFAULT_MASK_POLYGONIZATION_SMOOTHING_BUFFER).Buffer(-constants.DEFAULT_MASK_POLYGONIZATION_SMOOTHING_BUFFER)
                feature.SetGeometry(smoothed_geom)
                mask_lyr.SetFeature(feature)  # Save changes
        rds = None

def import_vector(src_dataset=None, src_layer=0, dst_dataset=None, dst_layer=None, access_mode=None, target_srs=None,
                  clip_dataset=None, clip_layer=None, return_handle=False,
                  **kwargs):
    """
    Import a vector layer into RAPIDA.
    Essentially this will reproject, crop and save to project GPKG
    :param src_dataset:
    :param src_layer:
    :param dst_dataset:
    :param dst_layer:
    :param access_mode:
    :param target_srs:
    :param clip_dataset:
    :param clip_layer:
    :param return_handle:
    :return:
    """
    if clip_dataset is not None:
        assert clip_layer is not None, f'clip_layer is required with clip_dataset'
    layer_creation_options = ['OVERWRITE=YES']
    with gdal.OpenEx(src_dataset, gdal.OF_VECTOR|gdal.OF_READONLY) as src_ds:
        try:
            int(src_layer)
            lyr = src_ds.GetLayer(src_layer)
        except ValueError:
            lyr = src_ds.GetLayerByName(src_layer)

        src_srs = lyr.GetSpatialRef()
        reproject = not proj_are_equal(src_srs, target_srs)
        translate_options = dict(
            format='GPKG', accessMode=access_mode, dstSRS=target_srs, reproject=reproject,
            layerName=dst_layer,  makeValid=True, clipDst=clip_dataset, clipDstLayer=clip_layer,
            preserveFID=True, geometryType='PROMOTE_TO_MULTI', layerCreationOptions=layer_creation_options,
            dim='XY'
        )
        translate_options.update(kwargs)
        imported_ds = gdal.VectorTranslate(destNameOrDestDS=dst_dataset, srcDS=src_ds, **translate_options)
        src_ds = None
        if not return_handle:
            imported_ds = None
        else:
            return imported_ds



def rasterize_vector_mask(vector_mask_ds=None, vector_mask_layer=None,
                          xres = constants.DEFAULT_MASK_RESOLUTION_METERS,
                          yres= constants.DEFAULT_MASK_RESOLUTION_METERS,
                          dst_dataset=None, nodata_value=None, **kwargs
                          ):

    rasterize_options = gdal.RasterizeOptions(
        format='GTiff', outputType=gdal.GDT_Byte,
        creationOptions=constants.GTIFF_CREATION_OPTIONS,
        noData=nodata_value, initValues=nodata_value,
        burnValues=1, layers=[vector_mask_layer],
        xRes=xres,
        yRes=yres,
        targetAlignedPixels=True,
        **kwargs


    )
    # rasterize_dict = dict(
    #         format='GTiff', outputType=gdal.GDT_Byte,
    #         creationOptions=constants.GTIFF_CREATION_OPTIONS,
    #         noData=nodata_value, initValues=nodata_value,
    #         burnValues=1, layers=[vector_mask_layer],
    #         xRes=xres,
    #         yRes=yres,
    #         targetAlignedPixels=True,
    # )

    rds = gdal.Rasterize(destNameOrDestDS=dst_dataset, srcDS=vector_mask_ds, options=rasterize_options)


def clip_raster_with_mask(source: str,
                          mask: str,
                          output_path: str,
                          mask_value=1,
                          nodata_value=None,
                          progress: Progress = None):
    """
    Clips the raster data at 'source' using the mask raster at 'mask' and saves the result to 'output_path'.

    :param source: Path to the source raster file.
    :param mask: Path to the mask raster file (e.g., project.raster_mask).
    :param output_path: File path to save the clipped raster.
    :param mask_value: Value to use for mask raster. Default is 1.
    :param nodata_value: Value to use for nodata raster. If None, nodata from source raster will be used.
    :return: output file path.
    """

    mask_task = None
    if progress:
        mask_task = progress.add_task(f"[green]Masking {source} by {mask}", total=100)

    # Define a callback function for GDAL that updates the Rich progress task.
    def progress_callback(complete, message, user_data):
        if progress and mask_task is not None:
            progress.update(mask_task, completed=int(complete * 100))
        return 1

    if nodata_value is not None:
        src_nodata = nodata_value
    else:
        with rasterio.open(source) as src:
            src_nodata = src.nodata

    with rasterio.open(mask) as mask_src:
        left, bottom, right, top = mask_src.bounds
        xRes, yRes = mask_src.res

    # Build a temporary VRT from the source raster to match the mask's grid.
    with tempfile.NamedTemporaryFile(suffix=".vrt") as tmp_vrt_file:
        vrt_filename = tmp_vrt_file.name

        with gdal.BuildVRT(vrt_filename, [source],
                           outputBounds=(left, bottom, right, top),
                           xRes=xRes,
                           yRes=yRes,
                           resampleAlg="nearest",
                           srcNodata=src_nodata
                           ) as vrt:

            calc_expr = f"(B=={mask_value})*A + (B!={mask_value})*{src_nodata}"

            calc_creation_options = {
                "COMPRESS": "ZSTD",
                "PREDICTOR": 2,
                "BIGTIFF": "IF_SAFER",
                "BLOCKXSIZE": "256",
                "BLOCKYSIZE": "256"
            }

            ds = Calc(
                calc=calc_expr,
                outfile=output_path,
                projectionCheck=True,
                format='GTiff',
                creation_options=calc_creation_options,
                overwrite=True,
                A=vrt,
                B=mask,
                NoDataValue=src_nodata,
                progress_callback=progress_callback
            )
            ds = None

    if progress and mask_task is not None:
        progress.remove_task(mask_task)

    return output_path