import logging
logger = logging.getLogger(__name__)
from osgeo import gdal, ogr
gdal.UseExceptions()



'''
ogr2ogr -sql "SELECT ST_PointOnSurface(geometry) as geometry, ogc_fid, area_in_meters, confidence FROM buildings" -dialect sqlite /tmp/bldgs1c.fgb /tmp/bldgs1.fgb
'''

def create_centroid(src_path=None, src_layer_name=None, dst_path=None, dst_srs=None):
    """
    Compute centroid for buildings layer
    :param src_path: str, path to input polygons
    :param src_layer_name, str, the layer name
    :param dst_path: str, path to out centroids
    :return: None
    """
    assert src_layer_name not in ('', None), f'invalid src_layer_name={src_layer_name}'

    options = dict(
        format='FlatGeobuf',
        SQLDialect='sqlite',
        SQLStatement=f'SELECT ST_PointOnSurface(geometry) as geometry, rowid as fid FROM {src_layer_name}',
        layerName=src_layer_name,
        geometryType='POINT'

    )
    if dst_srs is not None:
        options['dstSRS'] = dst_srs
        options['reproject'] = True

    options = gdal.VectorTranslateOptions(**options)


    ds = gdal.VectorTranslate(destNameOrDestDS=dst_path,srcDS=src_path,options=options)

    del ds




def buildings_in_mask_ogrio(buildings_centroid_path=None, mask_path=None, mask_pixel_value=None,
                            horizontal_chunks=None, vertical_chunks=None):
    """
    Select buildings whose centroid is inside the masked pixels
    :param buildings_centroid_path:
    :param mask_path:
    :param mask_pixel_value:
    :param horizontal_chunks:
    :param vertical_chunks:
    :return:
    """

    with gdal.Open(mask_path, gdal.OF_READONLY)  as mds:
        print(mds)



if __name__ == '__main__':
    import time
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(100)
    logging.basicConfig()
    logger.setLevel(logging.INFO)


    src_path = '/tmp/bldgs1.fgb'
    dst_path = '/tmp/bldgs1c.fgb'
    mask = '/data/surge/surge/stats/floods_mask.tif'


    start = time.time()


    create_centroid(src_path=src_path, src_layer_name='buildings', dst_path=dst_path, dst_srs='EPSG:3857')

    end = time.time()
    print((end-start))