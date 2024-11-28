import asyncio
import datetime
import json
from aiopmtiles import Reader
import morecantile as m
import mapbox_vector_tile
import itertools
from tqdm import tqdm
#import pandas as pd
import httpx
import logging
import concurrent
#from tit import atimeit
import itertools

GMOSM_BUILDINGS = 'https://data.source.coop/vida/google-microsoft-osm-open-buildings/pmtiles/goog_msft_osm.pmtiles'
GMOSM_BUILDINGS = 'https://data.source.coop/vida/google-microsoft-open-buildings/pmtiles/go_ms_building_footprints.pmtiles'
GMOSM_BUILDINGS = 'https://undpgeohub.blob.core.windows.net/userdata/9426cffc00b069908b2868935d1f3e90/datasets/bldgsc_20241029084831.fgb/bldgsc.pmtiles?sv=2025-01-05&ss=b&srt=o&se=2025-11-28T23%3A06%3A26Z&sp=r&sig=STScSa8aYPRaRDRjeMphS1UxJ7W2OACXGIYAT8k67YQ%3D'
WEB_MERCATOR_TMS = m.tms.get("WebMercatorQuad")

logger = logging.getLogger()

def chunker(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

#@atimeit
async def get_admin_level_bbox(iso3, admin_level=2):
    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = \
    f"""
        [out:json];
        area["ISO3166-1:alpha3"="{iso3}"][admin_level="{admin_level}"];
        rel(pivot)["boundary"="administrative"];
        out bb;
        
    """
    async with httpx.AsyncClient(timeout=100) as client:
        response = await client.post(overpass_url, data=overpass_query)
        data = response.json()
        print(response.text)
        bounds = data['elements'][0]['bounds']
        return bounds['minlon'], bounds['minlat'], bounds['maxlon'], bounds['maxlat']

# def pmtile2pandas(mvt_bytes=None,**kwargs):
#     """
#     Prints the human readable content of a chunk pof bytes representing a MVT tile(tiles)
#     :param mvt_bytes: bytes
#     :return:
#     """
#
#     decoded_data = mapbox_vector_tile.decode(mvt_bytes)
#     for lname, l in decoded_data.items():
#         records = []
#         for feat in l['features']:
#             props = feat['properties']
#             if kwargs:
#                 for k, v in kwargs.items():
#                     try:
#                         lv = props[k]
#                         if v == lv:
#                             records.append(props)
#                     except KeyError:
#                         pass
#             else:
#                 records.append(props)
#         if not records:
#             logger.debug(f'No recs were collected')
#             return
#         df = pd.DataFrame(records, columns=list(records[0].keys()))
#         return df
#

def dump_mvt(mvt_bytes=None,**kwargs):
    """
    Prints the human readable content of a chunk pof bytes representing a MVT tile(tiles)
    :param mvt_bytes: bytes
    :return:
    """

    decoded_data = mapbox_vector_tile.decode(mvt_bytes)
    for lname, l in decoded_data.items():
        logger.info(f'Dumping layer {lname}')
        for feat in l['features']:
            props = feat['properties']
            if kwargs:
                for k, v in kwargs.items():
                    if k in props:
                        lv = props[k]
                        if v == lv:
                            logger.info(f'{props}')
            logger.debug(f'{feat["id"]}, {feat["properties"]}')



#@atimeit
async def pmtiles2csv_par(url=None, iso3_country_code=None, **kwargs):
    start = datetime.datetime.now()
    async with Reader(url) as src:
        # PMTiles Metadata
        meta = await src.metadata()
        #logger.info(json.dumps(meta, indent=4))
        # Spatial Metadata
        bounds = src.bounds
        if iso3_country_code:
            bounds = await get_admin_level_bbox(iso3=iso3_country_code)
            logger.info(bounds)

        minzoom, maxzoom = src.minzoom, src.maxzoom
        zoom_level =  maxzoom//2+1


        # Is the data a Vector Tile Archive
        assert src.is_vector
        # bounds and original feature count
        west, south, east, north = bounds
        count = meta.get('tilestats')['layers'][0]['count']

        pdframes = list()
        if vector_layers := meta.get("vector_layers"):
            for layer in vector_layers:
                download_tasks = list()
                for i, tile in enumerate(tms.tiles(west=west, south=south, east=east, north=north,zooms=[zoom_level] ), start=1):
                    task = asyncio.create_task(
                        src.get_tile(z=tile.z, x=tile.x, y=tile.y),
                        name=f'{tile.z}/{tile.x}/{tile.y}'
                    )
                    download_tasks.append(task)
                done, pending = await asyncio.wait(download_tasks,timeout=1000, return_when=asyncio.ALL_COMPLETED)
                tiles_data = list()
                for dtask in done:
                    try:
                        tile_id = dtask.get_name()
                        tiles_data.append(await dtask)
                    except Exception as e:
                        logger.error(f'Failed to downlload tile {tile_id}')
                now = datetime.datetime.now()
                download_delta = now-start
                start_proc = datetime.datetime.now()
                pdframes = list()
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:

                    for chunk in chunker(tiles_data, size=50):
                        future_frames = [executor.submit(pmtile2pandas, mvt_bytes=e, **kwargs)  for e in chunk]

                        for f in concurrent.futures.as_completed(future_frames):
                            try:
                                tdf = f.result()
                                if tdf is not None:
                                    pdframes.append(tdf)
                            except Exception as e:
                                pass
                df = pd.concat(pdframes)
                df = df.drop_duplicates(subset=['adminid'], keep='first')
                nproc = len(df)
                processing_delta = datetime.datetime.now() - start_proc
                logger.info('#'*100)

                logger.info(f'{len(tiles_data)} tile(s) at zoom level {zoom_level} covering {bounds} were downloaded in {download_delta.total_seconds()} seconds')
                logger.info(f'Processing time was  {processing_delta.total_seconds()}')
                if count == nproc:
                    logger.info(f'All records {nproc} were fetched')
                else:
                    logger.info(f'{nproc} records were fetched')
                logger.info('#' * 100)

                df.to_csv('/data/hreaibm/admfieldmaps/a2pmtiles.csv')

def process_tile(data=None):
    ftrs = []
    try:
        decoded_data = mapbox_vector_tile.decode(data)
        for lname, l in decoded_data.items():
            logger.info(f'Dumping layer {lname}')
            for feat in l['features']:
                props = feat['properties']
                ftrs.append(feat['id'])
                logger.info(f'{feat["id"]} {props}')

    except Exception as e:
        logger.error(e)

    finally:
        return ftrs

def generator_length(gen):
    gen1, gen2 = itertools.tee(gen)
    length = sum(1 for _ in gen1)  # Consume the duplicate
    return length, gen2  # Return the length and the unconsumed generator

async def fetch_buildings(bbox=None, url=GMOSM_BUILDINGS):
    bb = m.BoundingBox(*bbox)
    assert WEB_MERCATOR_TMS.intersect_tms(bbox=bb) is True, f'The supplied bounding box {bb} does not intersect Web Mercator Quad '

    async with Reader(url) as src:
        # PMTiles Metadata
        meta = await src.metadata()
        #logger.info(json.dumps(meta, indent=4))
        # Spatial Metadata
        bounds = src.bounds
        print(bounds)
        bb = m.BoundingBox(*bounds)
        minzoom, maxzoom = src.minzoom, src.maxzoom
        zoom_level = maxzoom
        print(zoom_level)
        all_tiles = WEB_MERCATOR_TMS.tiles(west=bb.left,south=bb.bottom, east=bb.right, north=bb.top,
                                   zooms=[zoom_level],)
        ntiles, all_tiles = generator_length(all_tiles)
        with tqdm(total=ntiles, desc=f'Downloading...') as pbar:
            for i, tiles in enumerate(chunker(all_tiles, size=16), start=1):
                download_tasks = list()
                pbar.set_postfix_str(f'downloading chunk no {i}...', refresh=True)
                for tile in tiles :
                    task = asyncio.create_task(
                        src.get_tile(z=tile.z, x=tile.x, y=tile.y),
                        name=f'{tile.z}/{tile.x}/{tile.y}'
                    )
                    download_tasks.append(task)
                done, pending = await asyncio.wait(download_tasks, timeout=1000, return_when=asyncio.ALL_COMPLETED)
                tiles_data = list()
                for done_task in done:
                    try:
                        tile_id = done_task.get_name()
                        tiles_data.append(await done_task)
                    except Exception as e:
                        logger.error(f'Failed to download tile {tile_id}')
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:

                    future_frames = [executor.submit(process_tile, data=e) for e in tiles_data]

                    for f in concurrent.futures.as_completed(future_frames):
                        try:
                            r = f.result()
                            #logger.info(f'Chunk {i} produced ids {r}')
                            pbar.update(1)

                        except Exception as e:
                            logger.error(e)
                break
            pbar.set_postfix_str('Finished')


if __name__ == '__main__':
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(100)
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    url = 'https://undpgeohub.blob.core.windows.net/userdata/c75e50cd95568bafdc59dd731656044d/datasets/global_subset_20240424163643.geojson/global_subset_20240424163643.pmtiles?sv=2023-11-03&ss=b&srt=o&se=2025-04-24T17%3A07%3A29Z&sp=r&sig=t6Pc9SbT0oJIEynF8b1Gi4ENaQkEqlqyoVZYoSvXs3Y%3D'
    #asyncio.run(pmtiles2csv(url=url, iso3_country_code='KEN', iso3_count='KEN'))
    nf = 5829
    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    # bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    # bbox = 15.034157,49.282809,16.02842,49.66207 # CZE

    url='https://undpgeohub.blob.core.windows.net/userdata/4c9b6906ff63494418aef1efd028be44/datasets/global_ceei_final_2_20240618142311.gpkg/global_ceei.pmtiles?sv=2024-05-04&ss=b&srt=o&se=2025-06-21T13%3A14%3A58Z&sp=r&sig=lxRMTDofPiBFrIGuSPDQw1ubBCh3gcE9A6gf%2BtZ62GE%3D'
    url='https://undpgeohub.blob.core.windows.net/userdata/f7f7e1578d41891d14069e76958a692b/datasets/global_20240206174428.zip/global.pmtiles?sv=2024-05-04&ss=b&srt=o&se=2025-06-21T13%3A58%3A39Z&sp=r&sig=L9qUBm4ZEB%2BYVWUQKbDP8K%2FJtKpctroYGvVg8xfhE3U%3D'
    #asyncio.run(pmtiles2csv_par(url=url, iso3_country_code='KEN', iso3_count='KEN'))
    #asyncio.run(pmtiles2csv_par(url=url))
    #a =  asyncio.run(get_admin_level_bbox(iso3='ZWE'))
    #print(a)
    asyncio.run(fetch_buildings(bbox=bbox))
