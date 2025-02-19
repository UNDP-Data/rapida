
import random
import threading
from collections import deque
from cbsurge.components.builtenv.buildings.fgbgdal import OVERPASS_API_URL, GMOSM_BUILDINGS_ROOT
from pyogrio.raw import open_arrow
from cbsurge.components.builtenv.buildings.pmt import WEB_MERCATOR_TMS
from cbsurge.constants import ARROWTYPE2OGRTYPE
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
from rich.progress import Progress, TaskProgressColumn,BarColumn, TimeRemainingColumn, TextColumn
import httpx
from osm2geojson import json2geojson

from cbsurge.util.generator_length import generator_length
from cbsurge.util.http_post_json import http_post_json

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

            data = http_post_json(url=overpass_url, query=overpass_query, timeout=timeout)
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
                            #task.advance(nb)
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


def download_bbox(bbox=None, out_path=None, batch_size=5000, NWORKERS=4):
    """
    Download building from Google+Microsoft+OSM VIDA dataset based on bounds provided by
    equally sized blocks of space generated automatically in 3857 projection.


    :param out_path: str, abs path to buildings dataset
    :param bbox: iterable of floats, xmin, ymin, xmax,ymax
    :param batch_size: int, defaults to 5000, the number of buildings to download in one batch
    :param NWORKERS, int, defaults to 4. The number of threads to use for parallel download.
    :return:

    The implementation features several optimisations as to reduce the download:
        the number of countries that intersect the bbox are retrieved first from OSM
        a 3857 zoom level is computed from where tiles will ge generated in such a way that
        the average number of buildings in a tile would be roughly downloaded in one batch
        max NWORKERS tiles are downloaded in parallel

    the batch size can be used to control the size of a tile and the number of tiles used to download the
    buildings

    """
    assert batch_size>=1000, f'This is a small batch  size that will render the download inefficient'

    bb_poly = box(*bbox)

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

            batch_km2 = batch_size//country_bbox_buildings_density_km2
            for zoom_level, tile_area in ZOOMLEVEL_TILE_AREA.items():
                if tile_area<batch_km2:
                    sel_zom_level = zoom_level - 2 if zoom_level>2 else zoom_level
                    break

            all_tiles = WEB_MERCATOR_TMS.tiles( west=intersection_bb.left, south=intersection_bb.bottom,
                                                east=intersection_bb.right, north=intersection_bb.top,
                                                zooms=[sel_zom_level]
                                               )

            ntiles, all_tiles = generator_length(all_tiles)
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
                                    # filter out empty/invalid
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



def download_admin(admin_path=None, out_path=None, country_col_name=None, admin_col_name=None, batch_size=5000,
                   NWORKERS=4,
                   ):
    """
    Download building from Google+Microsoft+OSM VIDA dataset based on bounds provided by admin units


    :param admin_path: str, absolute path to a geospatial vector dataset representing administrative unit names
    :param out_path: str, abs path to buildings dataset
    :param country_col_name: str, name of the column from admin attr table that contains iso3 country code
    :param admin_col_name: str, name of the column from the admin attr table that contains amin unit name
    :param batch_size: int, defaults to 5000, the number of buildings to download in one batch
    :param NWORKERS, int, defaults to 4. The number of threads to use for parallel download.
    :return: None

    The buildings are downloaded in batches from country level disaggregated buildings dataset using  OGR Arrow protocol
    The buildings are downloaded in parallel using threads and double ended queues.
    NWORKERS specifies how many admin units are downloaded at the same time.

    """

    ndownloaded = 0
    failed = []
    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
        with ogr.Open(admin_path) as adm_ds:
            adm_lyr = adm_ds.GetLayer(0)
            all_features = [e for e in adm_lyr]
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
    from cbsurge.util.setup_logger import setup_logger
    logger = setup_logger(name='rapida', level=logging.INFO)

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
        #download_admin(admin_path=admin_path, out_path=out_path,
        #          country_col_name='shapeGroup', admin_col_name='shapeName', batch_size=3000)
        download_bbox(bbox=bbox, out_path=out_path, batch_size=5000 )
    except KeyboardInterrupt:
        pass

    end = time.time()
    print((end-start))