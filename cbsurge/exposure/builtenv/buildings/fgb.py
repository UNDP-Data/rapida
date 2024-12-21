import json
import random

import threading
import traceback
from collections import deque


from cbsurge.exposure.builtenv.buildings.fgbgdal import OVERPASS_API_URL, GMOSM_BUILDINGS_ROOT
from pyogrio.raw import open_arrow, write_arrow, read

from cbsurge.exposure.builtenv.buildings.pmt import WEB_MERCATOR_TMS
import morecantile as m
from cbsurge import util
import logging
import time
from pyogrio.core import read_info
from osgeo import ogr, osr, gdal
from shapely.geometry import box
from shapely import bounds
import shapely
from pyproj import Geod
import math
import concurrent
from pyarrow import compute as pc
from rich.progress import Progress, TaskProgressColumn,BarColumn,SpinnerColumn, TimeRemainingColumn, TextColumn
from rich.text import Text
import httpx
from osm2geojson import json2geojson

ogr.UseExceptions()
gdal.UseExceptions()
geod = Geod(ellps='WGS84')


class NumberColumn(TaskProgressColumn):
    def render(self, task):
        if task.total is not None:
            return super().render(task)
        else:
            return f"{task.completed} buildings"

progress_cols = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            NumberColumn(),
            TimeRemainingColumn()
            ]

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
#
#
# def add_fields_to_layer(layer=None, template_layer_info=None):
#
#     types = dict([(e[3:], getattr(ogr, e)) for e in dir(ogr) if e.startswith('OFT')])
#     for field_dict in template_layer_info['layers'][0]['fields']:
#         layer.CreateField(ogr.FieldDefn(field_dict['name'], types[field_dict['type']]))
#
#
# def read_bbox1(src_path=None, bbox=None):
#     with open_arrow(src_path, bbox=bbox, use_pyarrow=True) as source:
#         meta, reader = source
#         table = reader.read_all()
#         return meta, table
#
# def download_bbox(src_path=None, tile=None, dst_dir=None):
#     tile_bb = WEB_MERCATOR_TMS.bounds(tile)
#     tile_bbox = tile_bb.left, tile_bb.bottom, tile_bb.right, tile_bb.top
#     out_path = os.path.join(dst_dir, f'buildings_{tile.z}_{tile.x}_{tile.y}.fgb')
#     with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
#         with open_arrow(src_path, bbox=tile_bbox, use_pyarrow=True, batch_size=5000) as source:
#             meta, reader = source
#             fields = meta.pop('fields')
#             schema = reader.schema
#
#             if dst_ds.GetLayerCount() == 0:
#                 src_epsg = int(meta['crs'].split(':')[-1])
#                 src_srs = osr.SpatialReference()
#                 src_srs.ImportFromEPSG(src_epsg)
#                 dst_lyr = dst_ds.CreateLayer('buildings', geom_type=ogr.wkbPolygon, srs=src_srs)
#                 for name in schema.names:
#                     if 'wkb' in name or 'geometry' in name: continue
#                     field = schema.field(name)
#                     field_type = ARROWTYPE2OGRTYPE[field.type]
#                     dst_lyr.CreateField(ogr.FieldDefn(name, field_type))
#             logger.debug(f'Downloading buildings in batches from {src_path} {tile_bbox}')
#             for batch in reader:
#                 if batch.num_rows > 0:
#                     logger.debug(f'Writing {batch.num_rows} records in tile {tile}')
#                     dst_lyr.WritePyArrow(batch)
#         del dst_ds
#
#     nf = 0
#     with ogr.Open(out_path) as ds:
#         l = ds.GetLayer(0)
#         nf = l.GetFeatureCount()
#         logger.debug(f'{tile} - {nf}')
#         del ds
#     if nf == 0:
#         if os.path.exists(out_path):os.remove(out_path)
#         return
#     return out_path
#
#
# def download1(admin_path=None, out_path=None, country_col_name=None, admin_col_name=None, batch_size=5000 ):
#
#     with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
#         with ogr.Open(admin_path) as adm_ds:
#             adm_lyr = adm_ds.GetLayer(0)
#             all_features = [e for e in adm_lyr]
#             signal_event = threading.Event()
#             failed_tasks = []
#             ndownloaded = 0
#             #bar_format = "{percentage:3.0f}%|{bar}| downloaded {n_fmt} admin units out of {total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
#             with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#                 #with tqdm(total=len(all_features), desc=f'Downloaded', bar_format=bar_format, leave=False) as pbar:
#                 with Progress() as progress:
#                     total_task = progress.add_task(description=f'[red]Downloaded (0/{len(all_features)}) admin units', total=len(all_features))
#                     for features in util.chunker(all_features, size=4):
#                         futures = {}
#                         try:
#                             for feature in features:
#                                 au_geom = feature.GetGeometryRef()
#                                 au_poly = shapely.wkb.loads(bytes(au_geom.ExportToIsoWkb()))
#                                 props = feature.items()
#                                 country_iso3 = props[country_col_name]
#                                 au_name = props[admin_col_name]
#
#                                 remote_country_fgb_url = f'{GMOSM_BUILDINGS_ROOT}/country_iso={country_iso3}/{country_iso3}.fgb'
#                                 kw_args = dict(
#                                     src_path=remote_country_fgb_url,
#                                     mask=au_poly,
#                                     batch_size=batch_size,
#                                     signal_event=signal_event,
#                                     au_name=au_name,
#                                     progress=progress
#
#                                 )
#                                 futures[executor.submit(read_bbox, **kw_args)] = au_name
#
#                             #done, not_done = concurrent.futures.wait(futures, timeout=180 * len(futures), )
#
#                             for fut in concurrent.futures.as_completed(futures, timeout=180 * len(futures)):
#                                 au_name = futures[fut]
#                                 exception = fut.exception()
#                                 if exception is not None:
#                                     logger.error(f'Failed to process admin unit {au_name} because {exception} ')
#                                     failed_tasks.append((au_name, exception))
#                                 meta, batches = fut.result()
#                                 for batch in batches:
#                                     if dst_ds.GetLayerCount() == 0:
#                                         src_epsg = int(meta['crs'].split(':')[-1])
#                                         src_srs = osr.SpatialReference()
#                                         src_srs.ImportFromEPSG(src_epsg)
#                                         dst_lyr = dst_ds.CreateLayer('buildings', geom_type=ogr.wkbPolygon, srs=src_srs)
#                                         for name in batch.schema.names:
#                                             if 'wkb' in name or 'geometry' in name: continue
#                                             field = batch.schema.field(name)
#                                             field_type = ARROWTYPE2OGRTYPE[field.type]
#                                             dst_lyr.CreateField(ogr.FieldDefn(name, field_type))
#                                     #logger.info(f'Going to write {batch.num_rows} features from  {au_name} admin unit')
#                                     mask = pc.is_null(batch['wkb_geometry'])
#                                     should_filter = pc.any(mask).as_py()
#                                     if should_filter:
#                                         batch = batch.filter(mask)
#                                         for b in batch:
#                                             logger.info(b)
#                                     dst_lyr.WritePyArrow(batch)
#
#                                 #pbar.update()
#                                 assert dst_lyr.SyncToDisk() == 0
#                                 ndownloaded += 1
#                                 progress.update(total_task, description=f'[red]Downloaded ({ndownloaded}/{len(all_features)}) admin units',advance=1)
#                         except (concurrent.futures.CancelledError, KeyboardInterrupt) as pe:
#                             logger.info(f'Cancelling download. Please wait/allow for a graceful shutdown')
#                             signal_event.set()
#                             for fut, au_name in futures.items():
#                                 if fut.done():
#                                     logger.info(f'Finished downloading admin unit {au_name} ')
#                                     continue
#                                 c = fut.cancel() or not fut.done()
#                                 while not c:
#                                     c = fut.cancel() or fut.done()
#                                 logger.info(f'Building extraction in admin unit {au_name} was cancelled')
#                             if pe.__class__ == KeyboardInterrupt:
#                                 raise pe
#                             else:
#                                 break
#

#
# def download_pyogrio(bbox=None, out_path=None, batch_size:[int,None]=1000):
#     """
#     Download/stream buildings from VIDA buildings using pyogrio/pyarrow API
#
#     :param bbox: iterable of floats, xmin, ymin, xmax,ymax
#     :param out_path: str, full path where the buildings layer will be written
#     :param batch_size: int, default=1000, the max number of buildings to download in one batch
#     If supplied, the buildings are downloaded in batches otherwise they are streamd through pyarrow library
#     :return:
#     """
#
#
#     countries = get_countries_for_bbox_osm(bbox=bbox)
#     assert len(countries)> 0, f'The bounding box {bbox} does not intersect any country. Please make sure it makes sense!'
#
#     with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
#         for country in countries :
#             remote_country_fgb_url = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={country}/{country}.fgb'
#             if batch_size is not None:
#                 with open_arrow(remote_country_fgb_url, bbox=bbox, use_pyarrow=True, batch_size=batch_size) as source:
#                     meta, reader = source
#                     fields = meta.pop('fields')
#                     schema = reader.schema
#
#                     if dst_ds.GetLayerCount() == 0:
#                         src_epsg = int(meta['crs'].split(':')[-1])
#                         src_srs = osr.SpatialReference()
#                         src_srs.ImportFromEPSG(src_epsg)
#                         dst_lyr = dst_ds.CreateLayer('buildings', geom_type=ogr.wkbPolygon, srs=src_srs)
#                         for name in schema.names:
#                             if 'wkb' in name or 'geometry' in name:continue
#                             field = schema.field(name)
#                             field_type = ARROWTYPE2OGRTYPE[field.type]
#                             dst_lyr.CreateField(ogr.FieldDefn(name, field_type))
#                     logger.info(f'Downloading buildings in batches from {remote_country_fgb_url}')
#                     for batch in reader:
#                         logger.debug(f'Writing {batch.num_rows} records')
#                         dst_lyr.WritePyArrow(batch)
#             else:
#                 with open_arrow(remote_country_fgb_url, bbox=bbox, use_pyarrow=False) as source:
#                     meta, reader = source
#                     src_epsg = int(meta['crs'].split(':')[-1])
#                     src_srs = osr.SpatialReference()
#                     src_srs.ImportFromEPSG(src_epsg)
#                     logger.info(f'Streaming buildings from {remote_country_fgb_url}')
#                     write_arrow(reader, out_path,layer='buildings',driver='FlatGeobuf',append=True,
#                                 geometry_name='wkb_geometry', geometry_type='Polygon', crs=src_srs.ExportToWkt())
#     info = read_info(out_path, layer='buildings')
#     logger.info(f'{info["features"]} buildings were downloaded from {",".join(countries)} country datasets')
#
# def download_gdal(bbox=None, out_path=None, batch_size:[int, None]=1000):
#     """
#     Download/stream buildings from VIDA buildings using gdal/pyarrow API
#     :param bbox: iterable of floats, xmin, ymin, xmax,ymax
#     :param out_path: str, full path where the buildings layer will be written
#     :param batch_size: int, default=1000, the max number of buildings to be downloaded in one batch
#     If supplied, the buildings are downloaded in batches otherwise they are streamd through pyarrow library.
#     Batch downloading should be preferred in case of large bounding boxes/area
#
#     :return:
#     """
#
#
#     countries = get_countries_for_bbox_osm(bbox=bbox)
#     assert len(countries)> 0, f'The bounding box {bbox} does not intersect any country. Please make sure it makes sense!'
#     buildings = 0
#     with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
#
#         for country in get_countries_for_bbox_osm(bbox=bbox) :
#             remote_country_fgb_url = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={country}/{country}.fgb'
#             with ogr.Open(remote_country_fgb_url, gdal.OF_READONLY) as src_ds:
#                 src_lyr = src_ds.GetLayer(0)
#                 src_lyr.SetSpatialFilterRect(*bbox)
#                 if batch_size is not None:
#                     stream = src_lyr.GetArrowStream([f"MAX_FEATURES_IN_BATCH={batch_size}"])
#                 else:
#                     stream = src_lyr.GetArrowStream()
#                 schema = stream.GetSchema()
#                 if dst_ds.GetLayerCount() == 0:
#                     src_srs = src_lyr.GetSpatialRef()
#                     dst_lyr = dst_ds.CreateLayer('buildings', geom_type=ogr.wkbPolygon, srs=src_srs)
#                     for i in range(schema.GetChildrenCount()):
#                         if 'wkb' in schema.GetChild(i).GetName() or 'geometry' in schema.GetChild(i).GetName():continue
#                         dst_lyr.CreateFieldFromArrowSchema(schema.GetChild(i))
#                 if batch_size is not None:
#                     logger.info(f'Downloading buildings in batches from {remote_country_fgb_url}')
#                 else:
#                     logger.info(f'Streaming buildings from {remote_country_fgb_url}')
#                 while True:
#                     array = stream.GetNextRecordBatch()
#                     if array is None:
#                         break
#                     assert dst_lyr.WriteArrowBatch(schema, array) == ogr.OGRERR_NONE
#                     buildings+= array.GetLength()
#     logger.info(f'{buildings} buildings were downloaded from {",".join(countries)} country datasets')

def country_info(bbox=None, overpass_url=OVERPASS_API_URL):
    """
    Retrieves from overpass the geographic bounding box of the admin0 units corresponding to the
    iso3 country code
    :param iso3: ISO3 country code
    :param overpass_url: the overpass URL,
    :return: a tuple of numbers representing the geographical bbox as west, south, east, north
    """
    west, south, east, north = bbox
    overpass_query = \
        f"""
        [out:json][timeout:1800];
                                // Define the bounding box (replace {{bbox}} with "south,west,north,east").
                                (
                                relation["admin_level"="2"]["boundary"="administrative"]["type"="boundary"]({south}, {west}, {north}, {east});  // Match admin levels 0-10 // Ensure it's an administrative boundary

                                );

        /*added by auto repair*/
        (._;>;);
        /*end of auto repair*/
        out body;


    """

    overpass_query = f"""
                                   [out:json][timeout:1800];
                                    // Define the bounding box (replace {{bbox}} with "south,west,north,east").
                                    (
                                      relation
                                        ["admin_level"="2"]  // Match admin levels 0-10
                                        ["boundary"="administrative"] // Ensure it's an administrative boundary
                                        ["type"="boundary"]
                                        ({south}, {west}, {north}, {east});
                                    );

                                    // Output only polygons
                                    out geom;
            """


    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)

    try:

            data = util.http_post_json(url=overpass_url,data=overpass_query,timeout=timeout)
            countries = dict()
            geojson = json2geojson(data=data)

            for i, f in enumerate(geojson['features']):
                props = f['properties']
                tags = props.pop('tags')
                feature_geom = f['geometry']
                geom = shapely.geometry.shape(feature_geom)
                country_area, country_perimeter = geod.geometry_area_perimeter(geometry=geom)
                area_km2 = math.ceil(abs(country_area*1e-6))
                iso3_country_code = tags["ISO3166-1:alpha3"]
                countries[iso3_country_code] = area_km2, geom

            return countries

    except Exception as e:
        logger.error(f'Failed to fetch available countries from OSM. {e}')
        raise

def read_bbox(src_path=None, bbox=None, mask=None, batch_size=None, signal_event=None, name=None, ntries=3, progress=None):
    try:
        task = progress.add_task(description=f'[green]Downloading buildings in {name}...', start=False, total=None)
        for attempt in range(ntries):
            logger.debug(f'Attempt no {attempt} at {name}')
            try:
                with open_arrow(src_path, bbox=bbox, mask=mask, use_pyarrow=True, batch_size=batch_size) as source:
                    meta, reader = source
                    logger.debug(f'Opened {src_path}')
                    batches = []
                    nb = 0
                    for b in reader :
                        if signal_event.is_set():
                            #logger.info(f'{threading.current_thread().name} was signalled to stop')
                            logger.info(f'Cancelling building extraction in {name}')
                            return name, meta, batches
                        if b.num_rows > 0:
                            batches.append(b)
                            nb+=b.num_rows
                            progress.update(task, description=f'[green]Downloaded {nb} buildings in {name}', advance=nb, completed=None)
                            task.advance(nb)
                            #logger.info(f"Downloaded {nb:d} buildings in admin unit {au_name}[!n]")

                    return name, meta, batches
            except Exception as e:
                if attempt < ntries-1:
                    logger.info(f'Attempting to download {name} again')
                    time.sleep(1)
                    continue
                else:
                    return name, e, None

    finally:
        progress.remove_task(task)



def downloader(work=None, result=None, finished=None):


    logger.debug(f'starting building downloader thread {threading.current_thread().name}')
    while True:

        job = None
        try:
            job = work.pop()
        except IndexError as ie:
            pass
        if job is None:
            if finished.is_set():
                logger.debug(f'worker is finishing  in {threading.current_thread().name}')
                break
            continue

        if finished.is_set():
            break
        logger.debug(f'Starting job  {job["name"]}')
        result.append(read_bbox(**job))


def download_bbox(bbox=None, out_path=None, batch_size=5000):
    assert batch_size>=1000, f'This is a very small batch  size that will render the download inefficient'

    #bb = m.BoundingBox(*bbox)
    bb_poly = box(*bbox)

    NWORKERS = 4
    ndownloaded = 0
    countries = country_info(bbox=bbox)
    assert len(
        countries) > 0, f'The bounding box {bbox} does not intersect any country. Please make sure it makes sense!'
    cancelled = False
    failed = []
    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
        for country_iso3, country_data in countries.items():
            country_area_km2, country_geom = country_data
            remote_country_fgb_url = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={country_iso3}/{country_iso3}.fgb'
            info = read_info(remote_country_fgb_url)
            nfeatures_in_country = info['features']

            if not bb_poly.intersects(country_geom): #being paranoic
                logger.debug(f'{bbox} does not intersect {country_iso3}')
                continue
            country_bbox_buildings_density_km2 = nfeatures_in_country//country_area_km2
            sel_zom_level = None
            inters_pol =  bb_poly.intersection(country_geom)
            intersection_bbox = bounds(inters_pol)
            intersection_bb = m.BoundingBox(*intersection_bbox)
            intersection_area, _ = geod.geometry_area_perimeter(geometry=inters_pol)
            #intersection_area_km2 = math.ceil(abs(intersection_area*1e-6))
            #intersection_no_buildings = intersection_area_km2*country_bbox_buildings_density_km2
            batch_km2 = batch_size//country_bbox_buildings_density_km2
            for zoom_level, tile_area in ZOOMLEVEL_TILE_AREA.items():
                if tile_area<batch_km2:
                    sel_zom_level = zoom_level - 2 if zoom_level>2 else zoom_level
                    break



            all_tiles = WEB_MERCATOR_TMS.tiles( west=intersection_bb.left, south=intersection_bb.bottom,
                                                east=intersection_bb.right, north=intersection_bb.top,
                                                zooms=[sel_zom_level]
                                               )

            ntiles, all_tiles = util.generator_length(all_tiles)
            stop = threading.Event()
            jobs = deque()
            results = deque()

            logger.info(f'Downloading buildings from {country_iso3} in {ntiles} tiles/chunks generated in bbox {intersection_bbox} at zoom level {sel_zom_level} ')
            nworkers = ntiles if NWORKERS > ntiles else NWORKERS
            with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
                [executor.submit(downloader, jobs, results, stop) for i in range(nworkers)]
                with Progress(*progress_cols) as progress:
                    total_task = progress.add_task(
                        description=f'[red]Going to download buildings from {ntiles} tiles',
                        total=ntiles)

                    for tile in all_tiles:
                        tile_name = f'{tile.z}/{tile.x}/{tile.y}'
                        tile_bounds = WEB_MERCATOR_TMS.bounds(tile)
                        tile_bbox = tile_bounds.left,tile_bounds.bottom, tile_bounds.right, tile_bounds.top
                        tile_bb = box(*tile_bbox)
                        tile_intersection = tile_bb.intersection(inters_pol)
                        #tile_intersection_bbox = bounds(tile_intersection)

                        job = dict(
                            src_path=remote_country_fgb_url,
                            mask=tile_intersection,
                            batch_size=batch_size,
                            signal_event=stop,
                            name=tile_name,
                            progress=progress

                        )
                        jobs.append(job)

                    while True:
                        try:
                            try:
                                uname, meta, batches  = results.pop()
                                if batches is None:
                                    raise meta
                                logger.debug(f'{uname} was processed')
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
                                    lengths = pc.binary_length(batch.column("wkb_geometry"))
                                    mask = pc.greater(lengths, 0)
                                    should_filter = pc.any(pc.invert(mask)).as_py()
                                    if should_filter:
                                        batch = batch.filter(mask)
                                    if batch.num_rows == 0:
                                        logger.debug('skipping batch')
                                        continue
                                    try:
                                        dst_lyr.WritePyArrow(batch)
                                    except Exception as e:
                                        logger.info(f'batch with {batch.num_rows} rows from {uname} failed with error {e} and will be ignored')

                                dst_lyr.SyncToDisk()
                                ndownloaded += 1
                                progress.update(total_task,
                                                description=f'[red]Downloaded buildings from {ndownloaded} out of {ntiles} tiles',
                                                advance=1)
                            except IndexError as ie:
                                if not jobs and progress.finished:
                                    stop.set()
                                    break
                                s = random.random() # this one is necessary for ^C/KeyboardInterrupt
                                time.sleep(s)
                                continue
                        except Exception as e:
                            failed.append(f'Downloading {uname} failed: {e.__class__.__name__}("{e}")')
                            progress.update(total_task,
                                            description=f'[red]Downloaded buildings from {ndownloaded} out of {ntiles} tiles',
                                            advance=1)

                        except KeyboardInterrupt:
                            logger.info(f'Cancelling download. Please wait/allow for a graceful shutdown')
                            stop.set()
                            cancelled = True
                            break

            if cancelled:
                break

    if failed:
        for msg in failed:
            logger.error(msg)



def download_admin(admin_path=None, out_path=None, country_col_name=None, admin_col_name=None, batch_size=5000 ):
    """
    Download building from Google+Microsoft+OSM VIDA dataset
    :param admin_path:
    :param out_path:
    :param country_col_name:
    :param admin_col_name:
    :param batch_size:
    :return:
    """

    NWORKERS = 4
    ndownloaded = 0
    failed = []
    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
        with ogr.Open(admin_path) as adm_ds:
            adm_lyr = adm_ds.GetLayer(0)
            all_features = [e for e in adm_lyr]
            nfeatures = len(all_features)
            stop = threading.Event()
            jobs = deque()
            results = deque()
            with concurrent.futures.ThreadPoolExecutor(max_workers=NWORKERS) as executor:
                with Progress() as progress:
                    for feature in all_features:
                        au_geom = feature.GetGeometryRef()
                        au_poly = shapely.wkb.loads(bytes(au_geom.ExportToIsoWkb()))
                        props = feature.items()
                        country_iso3 = props[country_col_name]
                        au_name = props[admin_col_name]
                        remote_country_fgb_url = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={country_iso3}/{country_iso3}.fgb'
                        job = dict(
                            src_path=remote_country_fgb_url,
                            mask=au_poly,
                            batch_size=batch_size,
                            signal_event=stop,
                            name=au_name,
                            progress=progress

                        )
                        jobs.append(job)
                    njobs = len(jobs)
                    total_task = progress.add_task(
                        description=f'[red]Going to download buildings from {njobs} admin units', total=njobs)
                    nworkers = njobs if njobs < NWORKERS else NWORKERS
                    [executor.submit(downloader, jobs, results, stop) for i in range(nworkers)]
                    while True:
                        try:
                            try:
                                au_name, meta, batches  = results.pop()
                                if batches is None:
                                    raise meta
                                logger.debug(f'{au_name} was processed')
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

                                    lengths = pc.binary_length(batch.column("wkb_geometry"))
                                    mask = pc.greater(lengths, 0)
                                    should_filter = pc.any(pc.invert(mask)).as_py()
                                    if should_filter:
                                        batch = batch.filter(mask)
                                    if batch.num_rows ==0:
                                        logger.debug('skipping batch')
                                        continue
                                    try:
                                        dst_lyr.WritePyArrow(batch)
                                    except Exception as e:
                                        logger.info(f'batch with {batch.num_rows} rows from {au_name} failed with error {e} and will be ignored')


                                dst_lyr.SyncToDisk()
                                ndownloaded += 1
                                progress.update(total_task,
                                                description=f'[red]Downloaded buildings from {ndownloaded} out of {njobs} admin units',
                                                advance=1)
                            except IndexError as ie:
                                if not jobs and progress.finished:
                                    stop.set()
                                    break
                                s = random.random() # this one is necessary for ^C/KeyboardInterrupt
                                time.sleep(s)
                                continue

                        except Exception as e:
                            failed.append(f'Downloading {au_name} failed: {e.__class__.__name__}("{e}")')
                            ndownloaded += 1
                            progress.update(total_task,
                                            description=f'[red]Downloaded buildings from {ndownloaded} out of {njobs} admin units',
                                            advance=1)
                        except KeyboardInterrupt:
                            logger.info(f'Cancelling download. Please wait/allow for a graceful shutdown')
                            stop.set()
                            break


    if failed:
        for msg in failed:
            logger.error(msg)





if __name__ == '__main__':
    import asyncio
    logger = util.setup_logger(name='rapida', level=logging.INFO)

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
        # download_admin(admin_path=admin_path, out_path=out_path,
        #           country_col_name='shapeGroup', admin_col_name='shapeName', batch_size=3000)
        download_bbox(bbox=bbox, out_path=out_path, batch_size=5000 )
    except KeyboardInterrupt:
        pass

    end = time.time()
    print((end-start))