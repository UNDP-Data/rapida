from cbsurge import constants
from osgeo import gdal, ogr, osr
from cbsurge.util.proj_are_equal import proj_are_equal
import os
import logging
from functools import partial, wraps
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


def align_raster(target_srs=None, source=None, dst=None,
                 x_res: int = constants.DEFAULT_MASK_RESOLUTION_METERS,
                 y_res: int = constants.DEFAULT_MASK_RESOLUTION_METERS,
                 crop_ds=None, crop_layer_name=None, return_handle=False,
                 **kwargs
                 ):


    assert crop_layer_name not in ('', None), f'crop_layer_name: {crop_layer_name} is invalid'
    assert is_vector(crop_ds), f'{crop_ds} is not a vector dataset'

    should_crop = crop_ds and crop_layer_name

    warp_options = dict(
        format='GTiff',
        xRes=x_res,
        yRes=y_res,
        creationOptions=constants.GTIFF_CREATION_OPTIONS,
        cutlineDSName=crop_ds,
        cutlineLayer=crop_layer_name,
        cropToCutline=should_crop,
        targetAlignedPixels=True
    )
    warp_options.update(kwargs)

    with gdal.OpenEx(source, gdal.OF_RASTER | gdal.OF_READONLY) as src_ds:
        src_srs = src_ds.GetSpatialRef()
        prj_equal = proj_are_equal(src_srs, target_srs)
        if not prj_equal:
            # reproject raster mask to target projection
            logger.debug(f'Reprojecting {src_ds} {target_srs}')
            warp_options.update(dict(dstSRS=target_srs))
        rds = gdal.Warp(destNameOrDestDS=dst, srcDSOrSrcDSTab=src_ds, **warp_options)
        assert os.path.exists(dst), f'Failed to align {source}'
        if return_handle:
            return rds


def polygonize():

    with gdal.OpenEx(self.geopackage_file_path, gdal.OF_VECTOR | gdal.OF_UPDATE) as vds:
        logger.info(f'Polygonizing {raster_mask_local_path} ')
        mband = rds.GetRasterBand(1)
        mask_lyr = vds.CreateLayer('mask', geom_type=ogr.wkbMultiPolygon, srs=rds.GetSpatialRef())
        r = gdal.Polygonize(srcBand=mband, maskBand=mband,outLayer=mask_lyr,iPixValField=-1)

        assert r == 0, f'Failed to polygonize {raster_mask_local_path}'
        for feature in mask_lyr:
            geom = feature.GetGeometryRef()
            simplified_geom = geom.Simplify(constants.DEFAULT_MASK_POLYGONIZATION_SMOOTHING_BUFFER)  # Use SimplifyPreserveTopology(tolerance) if needed
            smoothed_geom = simplified_geom.Buffer(constants.DEFAULT_MASK_POLYGONIZATION_SMOOTHING_BUFFER).Buffer(-constants.DEFAULT_MASK_POLYGONIZATION_SMOOTHING_BUFFER)
            feature.SetGeometry(smoothed_geom)
            mask_lyr.SetFeature(feature)  # Save changes
        rds = None