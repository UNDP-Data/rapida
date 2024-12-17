import os.path
import tempfile

from cbsurge.exposure.builtenv.buildings.fgbgdal import get_countries_for_bbox_osm, GMOSM_BUILDINGS_ROOT
from pyogrio.raw import open_arrow, write_arrow
from cbsurge.exposure.builtenv.buildings.pmt import WEB_MERCATOR_TMS
import morecantile as m
from cbsurge import util
import logging
import time
from tqdm import tqdm
from pyogrio.core import read_info
from osgeo import ogr, osr, gdal
from shapely.geometry import box, shape
from shapely import bounds
from pyproj import Geod
import math
import flatgeobuf as fgbp

geod = Geod(ellps='WGS84')

logger = logging.getLogger(__name__)




async def download(admin_path=None, out_path=None, country_prop_name=None):

    with open(admin_path, mode='rb') as asrc:
        reader = fgbp.Reader(asrc)
        download_tasks = dict()
        failed_tasks = []
        features = [f for f in reader]
        bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt} tiles out of {total_fmt} [elapsed: {elapsed} remaining: {remaining}]"

        with tqdm(total=len(features), desc=f'Downloaded', bar_format=bar_format) as pbar:
            for i, units in enumerate(util.chunker(features, size=8), start=1):
                try:
                    for feature in units:
                        au_geom = shape(feature['geometry'])
                        au_bounds = au_geom.bounds

                        country_iso3 = feature['properties'][country_prop_name]
                        au_name = [v for k,v  in feature['properties'].items() if 'name' in k.lower()][0]

                        remote_country_fgb_url = f'{GMOSM_BUILDINGS_ROOT}/country_iso={country_iso3}/{country_iso3}.fgb'
                        task = asyncio.create_task(
                            fgbp.load_http_async(url=remote_country_fgb_url, bbox=au_bounds),
                            name=au_name
                        )
                        download_tasks[au_name] = task
                    print(len(download_tasks))
                    if not download_tasks:
                        logger.info(f'No data will be downlaoded')
                        return
                    done, pending = await asyncio.wait(download_tasks.values(), timeout=1000,
                                                       return_when=asyncio.ALL_COMPLETED)
                    print('haha')

                    for done_task in done:
                        try:
                            au_name = done_task.get_name()
                            au_data = await done_task
                            if au_data is not None:
                                logger.info(f'Writing {len(au_data["features"])} features for {au_name}')
                                pbar.update()
                            else:
                                logger.debug(f'No data was downloaded for tile {au_name} ')
                        except Exception as e:
                            logger.error(f'Failed to download tile {au_name}. {e}')
                            failed_tasks.append((au_name, e))
                    # handle pending
                    for pending_task in pending:
                        try:
                            au_name = pending_task.get_name()
                            pending_task.cancel()
                            await pending_task
                            failed_tasks.append((au_name, asyncio.Timeout))
                        except Exception as e:
                            logger.error(f'Failed to download buildings for admin unit {au_name}. {e}')
                            failed_tasks.append((au_name, e))
                except (asyncio.CancelledError, KeyboardInterrupt) as de:
                    logger.info(f'Cancelling download tasks')
                    for au_name, t in download_tasks.items():
                        if t.done(): continue
                        if not t.cancelled():
                            try:
                                t.cancel()
                                await t
                            except asyncio.CancelledError:
                                logger.debug(f'Download for admin  {au_name} was cancelled')
                    if de.__class__ == KeyboardInterrupt:
                        raise de
                    else:
                        break






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
    bbox = 90.62991666666666,20.739873437511918,92.35198706632379,24.836349986316765

    #a =  asyncio.run(get_admin_level_bbox(iso3='ZWE'))
    #print(a)

    out_path = '/tmp/bldgs1.fgb'
    admin_path = '/data/surge/surge/stats/adm3_flooded.fgb'
    # cntry = get_countries_for_bbox_osm(bbox=bbox)
    # print(cntry)
    start = time.time()

    asyncio.run(download(admin_path=admin_path,out_path=out_path, country_prop_name='shapeGroup'))
    end = time.time()
    print((end-start))