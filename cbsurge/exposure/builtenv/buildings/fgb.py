
import os.path
import tempfile
import threading

from cbsurge.exposure.builtenv.buildings.fgbgdal import get_countries_for_bbox_osm, GMOSM_BUILDINGS_ROOT
from pyogrio.raw import open_arrow, write_arrow, read

from cbsurge.exposure.builtenv.buildings.pmt import WEB_MERCATOR_TMS
import morecantile as m
from cbsurge import util
import logging
import time
from tqdm import tqdm
from pyogrio.core import read_info
from osgeo import ogr, osr, gdal
from shapely.geometry import box
from shapely import bounds
import shapely
from pyproj import Geod
import math
import concurrent
from tqdm.contrib.logging import logging_redirect_tqdm
ogr.UseExceptions()
gdal.UseExceptions()
geod = Geod(ellps='WGS84')


ZOOMLEVEL_TILE_AREA =  {
                        2: 60224669.5145855,
                        3: 21481830.870408278,
                        4: 4503416.3860813165,
                        5: 1008900.4679667788,
                        6: 237646.3339020322,
                        7: 57605.0263498855,
                        8: 14626.724809430663,
                        9: 3628.474442154846,
                        10: 903.5978962598876,
                        11: 225.4597166695404,
                        12: 56.309980584472655,
                        13: 14.084363345584869,
                        14: 3.5202322880802153,
                        15: 0.8799507586660384

                    }
logger = logging.getLogger(__name__)

ARROWTYPE2OGRTYPE = {'string':ogr.OFTString, 'double':ogr.OFTReal, 'int64':ogr.OFTInteger64, 'int':ogr.OFTInteger}
import pyarrow

def add_fields_to_layer(layer=None, template_layer_info=None):

    types = dict([(e[3:], getattr(ogr, e)) for e in dir(ogr) if e.startswith('OFT')])
    for field_dict in template_layer_info['layers'][0]['fields']:
        layer.CreateField(ogr.FieldDefn(field_dict['name'], types[field_dict['type']]))


def read_bbox1(src_path=None, bbox=None):
    with open_arrow(src_path, bbox=bbox, use_pyarrow=True) as source:
        meta, reader = source
        table = reader.read_all()
        return meta, table
def read_bbox(src_path=None,  mask=None, batch_size=None, signal_event=None, au_name=None, ntries=3):

    for attempt in range(ntries):
        try:

            with open_arrow(src_path,  mask=mask, use_pyarrow=True, batch_size=batch_size) as source:
                meta, reader = source
                batches = []
                nb = 0
                for b in reader :
                    if signal_event.is_set():
                        logger.debug(f'{threading.current_thread().name} was signalled to stop')
                        return
                    if b.num_rows > 0:
                        batches.append(b)
                        nb+=b.num_rows
                        logger.info(f"Downloaded {nb:d} buildings in admin unit {au_name}[!n]")

                return meta, batches
        except Exception as e:
            logger.error(e)
            if attempt == ntries-1:
                raise e
            time.sleep(1)
            logger.info(f'Attempting to download {au_name} again')





def download_bbox(src_path=None, tile=None, dst_dir=None):
    tile_bb = WEB_MERCATOR_TMS.bounds(tile)
    tile_bbox = tile_bb.left, tile_bb.bottom, tile_bb.right, tile_bb.top
    out_path = os.path.join(dst_dir, f'buildings_{tile.z}_{tile.x}_{tile.y}.fgb')
    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
        with open_arrow(src_path, bbox=tile_bbox, use_pyarrow=True, batch_size=5000) as source:
            meta, reader = source
            fields = meta.pop('fields')
            schema = reader.schema

            if dst_ds.GetLayerCount() == 0:
                src_epsg = int(meta['crs'].split(':')[-1])
                src_srs = osr.SpatialReference()
                src_srs.ImportFromEPSG(src_epsg)
                dst_lyr = dst_ds.CreateLayer('buildings', geom_type=ogr.wkbPolygon, srs=src_srs)
                for name in schema.names:
                    if 'wkb' in name or 'geometry' in name: continue
                    field = schema.field(name)
                    field_type = ARROWTYPE2OGRTYPE[field.type]
                    dst_lyr.CreateField(ogr.FieldDefn(name, field_type))
            logger.debug(f'Downloading buildings in batches from {src_path} {tile_bbox}')
            for batch in reader:
                if batch.num_rows > 0:
                    logger.debug(f'Writing {batch.num_rows} records in tile {tile}')
                    dst_lyr.WritePyArrow(batch)
        del dst_ds

    nf = 0
    with ogr.Open(out_path) as ds:
        l = ds.GetLayer(0)
        nf = l.GetFeatureCount()
        logger.debug(f'{tile} - {nf}')
        del ds
    if nf == 0:
        if os.path.exists(out_path):os.remove(out_path)
        return
    return out_path


def download(admin_path=None, out_path=None, country_col_name=None, admin_col_name=None, batch_size=5000 ):

    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
        with ogr.Open(admin_path) as adm_ds:
            adm_lyr = adm_ds.GetLayer(0)
            all_features = [e for e in adm_lyr]
            signal_event = threading.Event()
            failed_tasks = []
            bar_format = "{percentage:3.0f}%|{bar}| downloaded {n_fmt} admin units out of {total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                with tqdm(total=len(all_features), desc=f'Downloaded', bar_format=bar_format, leave=False) as pbar:
                    for features in util.chunker(all_features, size=4):
                        futures = {}

                        try:
                            for feature in features:
                                au_geom = feature.GetGeometryRef()
                                au_poly = shapely.wkb.loads(bytes(au_geom.ExportToIsoWkb()))
                                props = feature.items()
                                country_iso3 = props[country_col_name]
                                au_name = props[admin_col_name]

                                remote_country_fgb_url = f'{GMOSM_BUILDINGS_ROOT}/country_iso={country_iso3}/{country_iso3}.fgb'
                                kw_args = dict(
                                    src_path=remote_country_fgb_url,
                                    mask=au_poly,
                                    batch_size=batch_size,
                                    signal_event=signal_event,
                                    au_name=au_name,


                                )
                                futures[executor.submit(read_bbox, **kw_args)] = au_name

                            done, not_done = concurrent.futures.wait(futures, timeout=180 * len(futures), )

                            for fut in done:
                                au_name = futures[fut]
                                exception = fut.exception()
                                if exception is not None:
                                    logger.error(f'Failed to process tile {au_name} because {exception} ')
                                    failed_tasks.append((au_name, exception))
                                meta, batches = fut.result()
                                for batch in batches:
                                    if dst_ds.GetLayerCount() == 0:
                                        src_epsg = int(meta['crs'].split(':')[-1])
                                        src_srs = osr.SpatialReference()
                                        src_srs.ImportFromEPSG(src_epsg)
                                        dst_lyr = dst_ds.CreateLayer('buildings', geom_type=ogr.wkbPolygon, srs=src_srs)
                                        for name in batch.schema.names:
                                            if 'wkb' in name or 'geometry' in name: continue
                                            field = batch.schema.field(name)
                                            field_type = ARROWTYPE2OGRTYPE[field.type]
                                            dst_lyr.CreateField(ogr.FieldDefn(name, field_type))
                                    #logger.info(f'Going to write {batch.num_rows} features from  {au_name} admin unit')
                                    dst_lyr.WritePyArrow(batch)

                                pbar.update()
                                assert dst_lyr.SyncToDisk() == 0
                        except (concurrent.futures.CancelledError, KeyboardInterrupt) as pe:
                            logger.info(f'Cancelling download. Please wait/allow for a gracefull shutdown')
                            signal_event.set()

                            for fut, au_name in futures.items():
                                # while True:
                                #     print(f'{au_name} running: {fut.running()} done: {fut.done()} cancelled: {fut.cancelled()} cancel: {fut.cancel()}')
                                if fut.done():
                                    logger.info(f'{au_name} is done')
                                    continue
                                c = fut.cancel() or not fut.done()
                                while not c:
                                    c = fut.cancel() or fut.done()
                                logger.info(f'Building extraction in admin unit {au_name} was cancelled')
                            if pe.__class__ == KeyboardInterrupt:
                                raise pe
                            else:
                                break



def dpyogrio(bbox=None, out_path=None):

    #bb = m.BoundingBox(*bbox)
    bb_poly = box(*bbox)


    countries = get_countries_for_bbox_osm(bbox=bbox)
    assert len(
        countries) > 0, f'The bounding box {bbox} does not intersect any country. Please make sure it makes sense!'

    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
        with tempfile.TemporaryDirectory(delete=True) as temp_dir:
            for country in countries:
                remote_country_fgb_url = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={country}/{country}.fgb'
                info = read_info(remote_country_fgb_url)
                country_bounds = info['total_bounds']
                country_bbox = box(*country_bounds)
                if not bb_poly.intersects(country_bbox): #being paranoic
                    logger.debug(f'{bbox} does not intersect {remote_country_fgb_url} bbox {country_bbox}')
                    continue
                inters_pol =  bb_poly.intersection(country_bbox)
                intersection_bbox = bounds(inters_pol)
                intersection_bb = m.BoundingBox(*intersection_bbox)
                intersection_area, _ = geod.geometry_area_perimeter(geometry=inters_pol)
                intersection_area_km2 = math.ceil(abs(intersection_area*1e-6))
                sel_zom_level = None
                tile_area = None
                for zoom_level, tile_area in ZOOMLEVEL_TILE_AREA.items():
                    q, rem = divmod(intersection_area_km2, tile_area)
                    if q > 1e2:
                        sel_zom_level = zoom_level-1
                        tile_area = ZOOMLEVEL_TILE_AREA[sel_zom_level]
                        break
                all_tiles = WEB_MERCATOR_TMS.tiles( west=intersection_bb.left, south=intersection_bb.bottom,
                                                    east=intersection_bb.right, north=intersection_bb.top,
                                                    zooms=[sel_zom_level]
                                                   )
                bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt} tiles out of {total_fmt} - [elapsed: {elapsed} remaining: {remaining}]"
                ntiles, all_tiles = util.generator_length(all_tiles)

                tiles_path = []
                logger.info(f'Downloading buildings in {ntiles} tiles/chunks from {remote_country_fgb_url}')
                with tqdm(total=ntiles, desc=f'Downloaded', bar_format=bar_format) as pbar:
                    for i, tiles in enumerate(util.chunker(all_tiles, size=4), start=1):
                        futures = dict()
                        failed_tasks = []
                        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                            for tile in tiles:
                                tile_name = f'{tile.z}/{tile.x}/{tile.y}'
                                tile_bounds = WEB_MERCATOR_TMS.bounds(tile)
                                tile_bounds_poly = box(tile_bounds.left,tile_bounds.bottom, tile_bounds.right, tile_bounds.top)
                                ii = inters_pol.intersection(tile_bounds_poly)
                                iiarea = abs(geod.geometry_area_perimeter(ii)[0])*1e-6

                                kw_args = dict(src_path=remote_country_fgb_url, tile=tile, dst_dir=temp_dir)
                                futures[executor.submit(download_bbox, **kw_args)] = tile_name

                            done, not_done = concurrent.futures.wait(futures, timeout=160 * len(futures))
                            if not_done:
                                logger.info(f'Timeout issues')
                            for fut in done:
                                tile_name = futures[fut]
                                exception = fut.exception()
                                if exception is not None:
                                    #logger.error(f'Failed to process tile {tile_name} because {exception} ')
                                    failed_tasks.append((tile_name, exception))
                                    pbar.update(1)
                                    continue
                                tile_path = fut.result()
                                if tile_path:
                                    tiles_path.append(tile_path)
                                pbar.update(1)


                for i, input in enumerate(tiles_path):

                    ds = gdal.VectorTranslate(destNameOrDestDS=out_path, srcDS=input, accessMode='append',
                                              layerName='buildings')
                    del ds







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
    logger = util.setup_logger(name='rapida')

    nf = 5829
    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    #bbox = 19.5128619671,40.9857135911,19.5464217663,41.0120783699  # ALB, Divjake
    # bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    # bbox = 15.034157,49.282809,16.02842,49.66207 # CZE
    bbox = 19.350384,41.206737,20.059003,41.571459 # ALB, TIRANA
    bbox = 19.726666,39.312705,20.627545,39.869353, # ALB/GRC
    bbox = 90.62991666666666,20.739873437511918,92.35198706632379,24.836349986316765

    #a =  asyncio.run(get_admin_level_bbox(iso3='ZWE'))

    out_path = '/tmp/bldgs1.fgb'
    admin_path = '/data/surge/surge/stats/adm3_flooded.fgb'
    start = time.time()

    #asyncio.run(download(bbox=bbox))
    try:
        download(admin_path=admin_path, out_path=out_path,
                 country_col_name='shapeGroup', admin_col_name='shapeName', batch_size=3000)
    except KeyboardInterrupt:
        pass
    #dpyogrio(bbox=bbox, out_path=out_path, )
    end = time.time()
    print((end-start))