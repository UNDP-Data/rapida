import asyncio
import datetime
import gzip
import json
from typing import OrderedDict

from aiopmtiles import Reader
import morecantile as m
import mapbox_vector_tile
import itertools
from tqdm import tqdm
#import pandas as pd
import httpx
import logging
import concurrent
from osgeo import ogr, osr
from shapely.geometry import shape, MultiPolygon
from shapely.io import to_wkb
#from tit import atimeit
ogr.UseExceptions()
import itertools
from pyproj import Transformer
wm2ll_transformer = Transformer.from_crs(crs_from=3857, crs_to=4326)
import pyarrow as pa
GMOSM_BUILDINGS = 'https://data.source.coop/vida/google-microsoft-osm-open-buildings/pmtiles/goog_msft_osm.pmtiles'
#GMOSM_BUILDINGS = 'https://data.source.coop/vida/google-microsoft-open-buildings/pmtiles/go_ms_building_footprints.pmtiles'

WEB_MERCATOR_TMS = m.tms.get("WebMercatorQuad")
ll = osr.SpatialReference()
ll.ImportFromEPSG(4326)
logger = logging.getLogger()


def geoarrow_schema_adapter(schema: pa.Schema) -> pa.Schema:
    """
    Convert a geoarrow-compatible schema to a proper geoarrow schema

    This assumes there is a single "geometry" column with WKB formatting

    Parameters
    ----------
    schema: pa.Schema

    Returns
    -------
    pa.Schema
    A copy of the input schema with the geometry field replaced with
    a new one with the proper geoarrow ARROW:extension metadata

    """
    geometry_field_index = schema.get_field_index("geometry")
    geometry_field = schema.field(geometry_field_index)
    geoarrow_geometry_field = geometry_field.with_metadata(
        {b"ARROW:extension:name": b"geoarrow.wkb"}
    )

    geoarrow_schema = schema.set(geometry_field_index, geoarrow_geometry_field)

    return geoarrow_schema

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
#
# def dump_mvt(mvt_bytes=None,  **kwargs):
#     """
#     Prints the human readable content of a chunk pof bytes representing a MVT tile(tiles)
#     :param mvt_bytes: bytes
#     :return:
#     """
#
#     decoded_data = mapbox_vector_tile.decode(mvt_bytes)
#     for lname, l in decoded_data.items():
#         logger.info(f'Dumping layer {lname}')
#         for feat in l['features']:
#             props = feat['properties']
#             if kwargs:
#                 for k, v in kwargs.items():
#                     if k in props:
#                         lv = props[k]
#                         if v == lv:
#                             logger.info(f'{props}')
#             logger.debug(f'{feat["id"]}, {feat["properties"]}')
#

#
# #@atimeit
# async def pmtiles2csv_par(url=None, iso3_country_code=None, **kwargs):
#     start = datetime.datetime.now()
#     async with Reader(url) as src:
#         # PMTiles Metadata
#         meta = await src.metadata()
#         logger.info(json.dumps(meta, indent=4))
#         # Spatial Metadata
#         bounds = src.bounds
#         if iso3_country_code:
#             bounds = await get_admin_level_bbox(iso3=iso3_country_code)
#             logger.info(bounds)
#
#         minzoom, maxzoom = src.minzoom, src.maxzoom
#         zoom_level =  maxzoom//2+1
#
#
#         # Is the data a Vector Tile Archive
#         assert src.is_vector
#         # bounds and original feature count
#         west, south, east, north = bounds
#         count = meta.get('tilestats')['layers'][0]['count']
#
#         pdframes = list()
#         if vector_layers := meta.get("vector_layers"):
#             for layer in vector_layers:
#                 download_tasks = list()
#                 for i, tile in enumerate(tms.tiles(west=west, south=south, east=east, north=north,zooms=[zoom_level] ), start=1):
#                     task = asyncio.create_task(
#                         src.get_tile(z=tile.z, x=tile.x, y=tile.y),
#                         name=f'{tile.z}/{tile.x}/{tile.y}'
#                     )
#                     download_tasks.append(task)
#                 done, pending = await asyncio.wait(download_tasks,timeout=1000, return_when=asyncio.ALL_COMPLETED)
#                 tiles_data = list()
#                 for dtask in done:
#                     try:
#                         tile_id = dtask.get_name()
#                         tiles_data.append(await dtask)
#                     except Exception as e:
#                         logger.error(f'Failed to downlload tile {tile_id}')
#                 now = datetime.datetime.now()
#                 download_delta = now-start
#                 start_proc = datetime.datetime.now()
#                 pdframes = list()
#                 with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#
#                     for chunk in chunker(tiles_data, size=50):
#                         future_frames = [executor.submit(pmtile2pandas, mvt_bytes=e, **kwargs)  for e in chunk]
#
#                         for f in concurrent.futures.as_completed(future_frames):
#                             try:
#                                 tdf = f.result()
#                                 if tdf is not None:
#                                     pdframes.append(tdf)
#                             except Exception as e:
#                                 pass
#                 df = pd.concat(pdframes)
#                 df = df.drop_duplicates(subset=['adminid'], keep='first')
#                 nproc = len(df)
#                 processing_delta = datetime.datetime.now() - start_proc
#                 logger.info('#'*100)
#
#                 logger.info(f'{len(tiles_data)} tile(s) at zoom level {zoom_level} covering {bounds} were downloaded in {download_delta.total_seconds()} seconds')
#                 logger.info(f'Processing time was  {processing_delta.total_seconds()}')
#                 if count == nproc:
#                     logger.info(f'All records {nproc} were fetched')
#                 else:
#                     logger.info(f'{nproc} records were fetched')
#                 logger.info('#' * 100)
#
#                 df.to_csv('/data/hreaibm/admfieldmaps/a2pmtiles.csv')

def process_tile(data=None, compression=None, fields=None, tile_bounds=None):
    try:
        flds = {}
        for k in fields:
            flds[k] = []
        flds['wkb_geometry'] = []
        if compression is not None:
            data = gzip.decompress(data)
        #decoded_data = mapbox_vector_tile.decode(tile=data, default_options={'transformer':wm2ll_transformer.transform} )
        decoded_data = mapbox_vector_tile.decode(tile=data,  )
        for lname, l in decoded_data.items():
            for i, feat in enumerate(l['features']):
                fgeom = shape(feat['geometry'])
                print(fgeom)
                if fgeom.geom_type != 'Polygon':
                    continue
                props = feat['properties']
                for  k in fields:
                    flds[k].append(props.get(k, None))
                flds['wkb_geometry'].append(to_wkb(fgeom))
                break
            r = pa.RecordBatch.from_pydict(flds)
            #rr = r.replace_schema_metadata(metadata={'ARROW:extension:name':'geoarrow:wkb'})
            return  r


    except Exception as e:
        logger.error(e)

def generator_length(gen):
    gen1, gen2 = itertools.tee(gen)
    length = sum(1 for _ in gen1)  # Consume the duplicate
    return length, gen2  # Return the length and the unconsumed generator

def add_fields_to_layer(fields, layer):
    """
    Adds fields to the provided OGR layer based on the input dictionary.

    Args:
        fields (dict): A dictionary where keys are field names and values are field types.
        layer (ogr.Layer): The OGR layer to which the fields will be added.
    """
    # Define a mapping between field types in the dictionary and OGR field types
    field_type_map = {
        "Number": ogr.OFTReal,    # Number can be of type Real (float)
        "String": ogr.OFTString,  # String for textual data
    }

    # Loop through the fields dictionary and add each field to the layer
    for field_name, field_type in fields.items():
        # Check if field type exists in the map
        if field_type not in field_type_map:
            raise ValueError(f"Unsupported field type: {field_type} for field '{field_name}'")

        # Get the OGR field type from the map
        ogr_field_type = field_type_map[field_type]

        # Add field to the layer
        layer.CreateField(ogr.FieldDefn(field_name, ogr_field_type))


async def fetch_buildings(bbox=None, zoom_level=None, url=GMOSM_BUILDINGS):
    bb = m.BoundingBox(*bbox)
    assert WEB_MERCATOR_TMS.intersect_tms(bbox=bb) is True, f'The supplied bounding box {bb} does not intersect Web Mercator Quad '

    async with Reader(url) as src:
        # PMTiles Metadata
        header = src._header
        compression = header['tile_compression'] if header['tile_compression'] is not None else None

        meta = await src.metadata()
        layer_meta = meta['vector_layers'][0]


        # Spatial Metadata
        bounds = src.bounds
        # print(bounds)
        # bb = m.BoundingBox(*bounds)
        minzoom, maxzoom = src.minzoom, src.maxzoom
        zoom_level = zoom_level or maxzoom

        all_tiles = WEB_MERCATOR_TMS.tiles(west=bb.left,south=bb.bottom, east=bb.right, north=bb.top,
                                   zooms=[zoom_level],)
        ntiles, all_tiles = generator_length(all_tiles)

        with ogr.GetDriverByName('FlatGeobuf').CreateDataSource('/tmp/bldgs.fgb') as dst_ds:
            dst_lyr = dst_ds.CreateLayer('bldgs', geom_type=ogr.wkbPolygon, srs=ll)
            add_fields_to_layer(fields=layer_meta['fields'],layer=dst_lyr)

            stream= dst_lyr.GetArrowStream()
            schema = stream.GetSchema()

            with tqdm(total=ntiles, desc=f'Downloading...') as pbar:
                for i, tiles in enumerate(chunker(all_tiles, size=1), start=1):

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
                            tile_data = await done_task
                            if tile_data is not None:
                                tiles_data.append(tile_data)
                            else:
                                logger.info(f'No data was downloaded for {tile_id} ')
                        except Exception as e:
                            logger.error(f'Failed to download tile {tile_id}')
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        futures = [executor.submit(process_tile, data=e, compression=compression, fields=layer_meta['fields']) for e in tiles_data]
                        done, not_done = concurrent.futures.wait(futures,timeout=300, )
                        for fut in done:
                            arr = fut.result()
                            dst_lyr.WritePyArrow(arr)
                    break
                pbar.set_postfix_str('Finished')


if __name__ == '__main__':
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(100)
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    nf = 5829
    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    # bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    # bbox = 15.034157,49.282809,16.02842,49.66207 # CZE

    #a =  asyncio.run(get_admin_level_bbox(iso3='ZWE'))
    #print(a)
    url = 'https://undpgeohub.blob.core.windows.net/userdata/9426cffc00b069908b2868935d1f3e90/datasets/bldgsc_20241029084831.fgb/bldgsc.pmtiles?sv=2025-01-05&ss=b&srt=o&se=2025-11-29T15%3A58%3A37Z&sp=r&sig=bQ8pXRRkNqdsJbxcIZ1S596u4ZvFwmQF3TJURt3jSP0%3D'
    asyncio.run(fetch_buildings(bbox=bbox, zoom_level=None))
