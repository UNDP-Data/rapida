import json
import os.path

from cbsurge.exposure.builtenv.buildings.fgbgdal import get_countries_for_bbox_osm, GMOSM_BUILDINGS_ROOT
from pyogrio.raw import open_arrow, write_arrow
import logging
import time
from tqdm import tqdm
from pyogrio.core import read_info
from osgeo import ogr, osr, gdal




logger = logging.getLogger(__name__)

ARROWTYPE2OGRTYPE = {'string':ogr.OFTString, 'double':ogr.OFTReal, 'int64':ogr.OFTInteger64, 'int':ogr.OFTInteger}


def add_fields_to_layer(layer=None, template_layer_info=None):

    types = dict([(e[3:], getattr(ogr, e)) for e in dir(ogr) if e.startswith('OFT')])
    for field_dict in template_layer_info['layers'][0]['fields']:
        layer.CreateField(ogr.FieldDefn(field_dict['name'], types[field_dict['type']]))


def download_pyogrio(bbox=None, out_path=None, batch_size:[int,None]=1000):
    """
    Download/stream buildings from VIDA buildings using pyogrio/pyarrow API

    :param bbox: iterable of floats, xmin, ymin, xmax,ymax
    :param out_path: str, full path where the buildings layer will be written
    :param batch_size: int, default=1000, the max number of buildings to download in one batch
    If supplied, the buildings are downloaded in batches otherwise they are streamd through pyarrow library
    :return:
    """


    countries = get_countries_for_bbox_osm(bbox=bbox)
    assert len(countries)> 0, f'The bounding box {bbox} does not intersect any country. Please make sure it makes sense!'

    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
        for country in countries :
            remote_country_fgb_url = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={country}/{country}.fgb'
            if batch_size is not None:
                with open_arrow(remote_country_fgb_url, bbox=bbox, use_pyarrow=True, batch_size=batch_size) as source:
                    meta, reader = source
                    fields = meta.pop('fields')
                    schema = reader.schema

                    if dst_ds.GetLayerCount() == 0:
                        src_epsg = int(meta['crs'].split(':')[-1])
                        src_srs = osr.SpatialReference()
                        src_srs.ImportFromEPSG(src_epsg)
                        dst_lyr = dst_ds.CreateLayer('buildings', geom_type=ogr.wkbPolygon, srs=src_srs)
                        for name in schema.names:
                            if 'wkb' in name or 'geometry' in name:continue
                            field = schema.field(name)
                            field_type = ARROWTYPE2OGRTYPE[field.type]
                            dst_lyr.CreateField(ogr.FieldDefn(name, field_type))
                    logger.info(f'Downloading buildings in batches from {remote_country_fgb_url}')
                    for batch in reader:
                        logger.debug(f'Writing {batch.num_rows} records')
                        dst_lyr.WritePyArrow(batch)
            else:
                with open_arrow(remote_country_fgb_url, bbox=bbox, use_pyarrow=False) as source:
                    meta, reader = source
                    src_epsg = int(meta['crs'].split(':')[-1])
                    src_srs = osr.SpatialReference()
                    src_srs.ImportFromEPSG(src_epsg)
                    logger.info(f'Streaming buildings from {remote_country_fgb_url}')
                    write_arrow(reader, out_path,layer='buildings',driver='FlatGeobuf',append=True,
                                geometry_name='wkb_geometry', geometry_type='Polygon', crs=src_srs.ExportToWkt())
    info = read_info(out_path, layer='buildings')
    logger.info(f'{info["features"]} buildings were downloaded from {",".join(countries)} country datasets')



def download_gdal(bbox=None, out_path=None, batch_size:[int, None]=1000):
    """
    Download/stream buildings from VIDA buildings using gdal/pyarrow API
    :param bbox: iterable of floats, xmin, ymin, xmax,ymax
    :param out_path: str, full path where the buildings layer will be written
    :param batch_size: int, default=1000, the max number of buildings to be downloaded in one batch
    If supplied, the buildings are downloaded in batches otherwise they are streamd through pyarrow library.
    Batch downloading should be preferred in case of large bounding boxes/area

    :return:
    """


    countries = get_countries_for_bbox_osm(bbox=bbox)
    assert len(countries)> 0, f'The bounding box {bbox} does not intersect any country. Please make sure it makes sense!'
    buildings = 0
    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:

        for country in get_countries_for_bbox_osm(bbox=bbox) :
            remote_country_fgb_url = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={country}/{country}.fgb'
            with ogr.Open(remote_country_fgb_url, gdal.OF_READONLY) as src_ds:
                src_lyr = src_ds.GetLayer(0)
                src_lyr.SetSpatialFilterRect(*bbox)
                if batch_size is not None:
                    stream = src_lyr.GetArrowStream([f"MAX_FEATURES_IN_BATCH={batch_size}"])
                else:
                    stream = src_lyr.GetArrowStream()
                schema = stream.GetSchema()
                if dst_ds.GetLayerCount() == 0:
                    src_srs = src_lyr.GetSpatialRef()
                    dst_lyr = dst_ds.CreateLayer('buildings', geom_type=ogr.wkbPolygon, srs=src_srs)
                    for i in range(schema.GetChildrenCount()):
                        if 'wkb' in schema.GetChild(i).GetName() or 'geometry' in schema.GetChild(i).GetName():continue
                        dst_lyr.CreateFieldFromArrowSchema(schema.GetChild(i))
                if batch_size is not None:
                    logger.info(f'Downloading buildings in batches from {remote_country_fgb_url}')
                else:
                    logger.info(f'Streaming buildings from {remote_country_fgb_url}')
                while True:
                    array = stream.GetNextRecordBatch()
                    if array is None:
                        break
                    assert dst_lyr.WriteArrowBatch(schema, array) == ogr.OGRERR_NONE
                    buildings+= array.GetLength()
    logger.info(f'{buildings} buildings were downloaded from {",".join(countries)} country datasets')






if __name__ == '__main__':
    import asyncio
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(100)
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    nf = 5829
    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    #bbox = 19.5128619671,40.9857135911,19.5464217663,41.0120783699  # ALB, Divjake
    # bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    # bbox = 15.034157,49.282809,16.02842,49.66207 # CZE
    bbox = 19.350384,41.206737,20.059003,41.571459 # ALB, TIRANA
    bbox = 19.726666,39.312705,20.627545,39.869353, # ALB/GRC

    #a =  asyncio.run(get_admin_level_bbox(iso3='ZWE'))
    #print(a)
    url = 'https://undpgeohub.blob.core.windows.net/userdata/9426cffc00b069908b2868935d1f3e90/datasets/bldgsc_20241029084831.fgb/bldgsc.pmtiles?sv=2025-01-05&ss=b&srt=o&se=2025-11-29T15%3A58%3A37Z&sp=r&sig=bQ8pXRRkNqdsJbxcIZ1S596u4ZvFwmQF3TJURt3jSP0%3D'
    #validate_source()
    out_path = '/tmp/bldgs1.fgb'
    # cntry = get_countries_for_bbox_osm(bbox=bbox)
    # print(cntry)
    start = time.time()
    #asyncio.run(download(bbox=bbox))

    download_gdal(bbox=bbox, out_path=out_path, batch_size=3000)

    end = time.time()
    print((end-start))