from cbsurge import constants
from osgeo import gdal, ogr, osr
from cbsurge.util.proj_are_equal import proj_are_equal
import os
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


def import_raster(source=None, dst=None, target_srs=None,
                  x_res: int = constants.DEFAULT_MASK_RESOLUTION_METERS,
                  y_res: int = constants.DEFAULT_MASK_RESOLUTION_METERS,
                  crop_ds=None, crop_layer_name=None, return_handle=False,
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

    should_crop = crop_ds is not None and crop_layer_name not in ('', None)


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
                  clip_dataset=None, clip_layer=None, return_handle=False):
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
            preserveFID=True, geometryType='PROMOTE_TO_MULTI', layerCreationOptions=layer_creation_options
        )
        imported_ds = gdal.VectorTranslate(destNameOrDestDS=dst_dataset, srcDS=src_ds, **translate_options)
        src_ds = None
        if not return_handle:
            imported_ds = None
        else:
            return imported_ds



def rasterize_vector_mask(src_dataset=None, src_layer=0,
                          xres = constants.DEFAULT_MASK_RESOLUTION_METERS,
                          yres= constants.DEFAULT_MASK_RESOLUTION_METERS,
                          dst_dataset=None, nodata_value=None
                          ):


    rasterize_options = gdal.RasterizeOptions(
        format='GTiff', outputType=gdal.GDT_Byte,
        creationOptions=constants.GTIFF_CREATION_OPTIONS,
        noData=nodata_value, initValues=nodata_value,
        burnValues=1, layers=[src_layer],
        xRes=xres,
        yRes=yres,
        targetAlignedPixels=True,


    )
    rds = gdal.Rasterize(destNameOrDestDS=dst_dataset, srcDS=src_dataset, options=rasterize_options)