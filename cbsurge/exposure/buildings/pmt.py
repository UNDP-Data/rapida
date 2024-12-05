import asyncio
import gzip
import json
import math
import h3.api.basic_int as h3
import os.path
from functools import partial
from aiopmtiles import Reader
import morecantile as m
import mapbox_vector_tile
from tqdm import tqdm
import httpx
import logging
import concurrent
from osgeo import ogr, osr
from shapely.geometry import shape, box
from shapely.io import to_wkb
from shapely import  transform
import numpy as np
import itertools
import pyarrow as pa
import pyarrow.compute as pc

from cbsurge.util import validate
from pyproj import Geod
ogr.UseExceptions()
osr.UseExceptions()
GMOSM_BUILDINGS = 'https://data.source.coop/vida/google-microsoft-osm-open-buildings/pmtiles/goog_msft_osm.pmtiles'


WEB_MERCATOR_TMS = m.tms.get("WebMercatorQuad")
WEB_MERCATOR_BBOX = WEB_MERCATOR_TMS.xy_bbox
WEB_MERCATOR_EXTENT = WEB_MERCATOR_BBOX.right - WEB_MERCATOR_BBOX.left
srs_prj = osr.SpatialReference()
srs_prj.ImportFromEPSG(4326)
logger = logging.getLogger(__name__)
geod = Geod(ellps="WGS84")

def validate_source():
    try:

        validate(url=GMOSM_BUILDINGS, timeout=3)
    except Exception as e:
        logger.error(f'Failed to validate source {GMOSM_BUILDINGS}. Error is {e}')

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


def process_tile(data=None, compression=None, fields=None, tile=None):
    try:
        flds = {}
        for k in fields:
            flds[k] = []
        flds['wkb_geometry'] = []
        if compression is not None:
            data = gzip.decompress(data)
        decoded_data = mapbox_vector_tile.decode(tile=data, default_options=dict(
            transformer=partial(mvt2ll1, z=tile.z,tx=tile.x,ty=tile.y),
            y_coord_down=True)
        )

        fcol = []
        for lname, l in decoded_data.items():
            fcol.extend(l['features'])
            for i, feat in enumerate(l['features']):
                fgeom = shape(feat['geometry'])
                if fgeom.geom_type == 'MultiPolygon':
                    polys = fgeom.geoms
                else:
                    polys = fgeom,
                props = feat['properties']
                for p in polys:
                    for  k, v in fields.items():
                        nv = props.get(k, None)
                        if nv is None:
                            if v == 'Number':
                                nv = np.nan
                            if v == 'String':
                                nv = ''
                        flds[k].append(nv)
                    flds['wkb_geometry'].append(to_wkb(p))

            r = pa.RecordBatch.from_pydict(flds)


            #logger.info(f'Tile {tile.z}/{tile.x}/{tile.y} contains {r.num_rows} polygons and {n_multi_geom} multi geoms')
            return  r
    except Exception as e:
        logger.error(e)
        raise
def process_tile_n(data=None, compression=None, fields=None, tile=None, bbox_poly=None):
    try:

        flds = {}
        for k in fields:
            flds[k] = []
        flds['wkb_geometry'] = []
        flds['location_in_tile'] = []
        flds['aream2'] = []

        if compression is not None:
            data = gzip.decompress(data)
        # decoded_data = mapbox_vector_tile.decode(tile=data, default_options=dict(
        #     transformer=partial(mvt2ll1, z=tile.z,tx=tile.x,ty=tile.y),
        #     y_coord_down=True)
        # )
        decoded_data1 = mapbox_vector_tile.decode(tile=data, default_options=dict(
            #transformer=partial(mvt2ll1, z=tile.z, tx=tile.x, ty=tile.y),
            y_coord_down=True))

        for lname, l in decoded_data1.items():
            for feat in l['features']:
                fgeom = shape(feat['geometry'])
                props = feat['properties']
                if fgeom.geom_type == 'MultiPolygon':
                    polys = fgeom.geoms
                else:
                    polys = fgeom,
                for p in polys:
                    c = np.array(p.exterior.coords, dtype='i2')
                    cs = c.size
                    tile_mask = (c>=0) & (c<=l['extent'])
                    buffer_mask = ~tile_mask
                    split_mask = (c==-80) | (c==l['extent']+80)
                    is_split = np.any(split_mask)
                    is_tile = tile_mask.sum()==cs
                    is_buffer = buffer_mask.sum() > 0
                    if is_tile:
                        btype = 'inside'
                    if is_buffer:
                        btype = 'buffer'
                    if is_split:
                        btype = 'edge'
                        continue

                    tp = transform(p, transformation=partial(mvt2ll, z=tile.z, x=tile.x, y=tile.y))
                    if not tp.within(bbox_poly):
                        continue
                    area_m = abs(geod.geometry_area_perimeter(tp)[0])
                    if area_m in flds['aream2']:
                        logger.debug(f'Polygon with area {area_m} will be skipped because it is a dup')
                        continue


                    for k, v in fields.items():
                        nv = props.get(k, None)
                        if nv is None:
                            if v == 'Number':
                                nv = np.nan
                            if v == 'String':
                                nv = ''

                        flds[k].append(nv)

                    flds['aream2'].append(area_m)
                    flds['location_in_tile'].append(btype)
                    #flds['h3id'].append(h3.latlng_to_cell(lat=tp.centroid.y, lng=tp.centroid.x, res=11))
                    flds['wkb_geometry'].append(to_wkb(tp))
            r = pa.Table.from_pydict(flds)

            return r

        #logger.info(f'Tile {tile.z}/{tile.x}/{tile.y} contains {r.num_rows} polygons and {n_multi_geom} multi geoms')
        #return  filtered_r


    except Exception as e:
        logger.error(e)
        raise



def generator_length(gen):
    gen1, gen2 = itertools.tee(gen)
    length = sum(1 for _ in gen1)  # Consume the duplicate
    return length, gen2  # Return the length and the unconsumed generator

def mvt2ll1(x=None, y=None, tx=None, ty=None, z=None, extent=4096):

    size = extent * 2 ** z
    x0 = extent * tx
    y0 = extent * ty
    lon = (x + x0) * 360. / size - 180
    y2 = 180 - (y + y0) * 360. / size
    lat = 360. / math.pi * math.atan(math.exp(y2 * np.pi / 180)) - 90
    return lon, lat

def mvt2ll(coords_array=None, x=None, y=None, z=None, extent=4096):

    px, py = coords_array[..., 0], coords_array[..., 1]
    size = extent*2**z
    x0 = extent*x
    y0 = extent*y
    lon = (px+x0)*360./size-180
    y2 = 180 - (py + y0) * 360. / size
    lat = 360. / np.pi * np.arctan(np.exp(y2 * np.pi / 180)) - 90
    tcoords = np.stack((lon, lat), axis=-1)
    return tcoords


def bbox2ogrgeom(bbox=None):


    # Create a rectangular geometry from the bounding box
    bbox_geom = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(bbox[0], bbox[1])  # min_x, min_y
    ring.AddPoint(bbox[2], bbox[1])  # max_x, min_y
    ring.AddPoint(bbox[2], bbox[3])  # max_x, max_y
    ring.AddPoint(bbox[0], bbox[3])  # min_x, max_y
    ring.AddPoint(bbox[0], bbox[1])  # Closing the ring
    bbox_geom.AddGeometry(ring)
    return bbox_geom

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
    layer.CreateField(ogr.FieldDefn('location_in_tile', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('aream2', ogr.OFTReal))



async def fetch_buildings(bbox=None, zoom_level=None, url=GMOSM_BUILDINGS):
    bb = m.BoundingBox(*bbox)
    bbox_polygon = box(*bbox)
    assert WEB_MERCATOR_TMS.intersect_tms(bbox=bb) is True, (f'The supplied bounding box {bb} does not intersect the '
                                                             f'Web Mercator Quad ')

    t = f'\n ogr2ogr -spat {bb.left} {bb.bottom} {bb.right} {bb.top} /tmp/bldgs_man.fgb "/vsicurl/https://data.source.coop/vida/google-microsoft-osm-open-buildings/flatgeobuf/by_country/country_iso=ALB/ALB.fgb"'
    # if tile.x != 18161 or tile.y != 12285:continue
    #logger.info(t)
    async with Reader(url) as src:
        # PMTiles Metadata
        header = src._header
        compression = header['tile_compression'] if header['tile_compression'] is not None else None

        meta = await src.metadata()
        layer_meta = meta['vector_layers'][0]
        logger.debug(json.dumps(layer_meta, indent=2))
        minzoom, maxzoom = src.minzoom, src.maxzoom
        zoom_level = zoom_level or maxzoom

        all_tiles = WEB_MERCATOR_TMS.tiles(west=bb.left,south=bb.bottom, east=bb.right, north=bb.top,
                                   zooms=[zoom_level],)
        ntiles, all_tiles = generator_length(all_tiles)

        with ogr.GetDriverByName('FlatGeobuf').CreateDataSource('/tmp/bldgs.fgb') as dst_ds:
            dst_lyr = dst_ds.CreateLayer('bldgs', geom_type=ogr.wkbPolygon, srs=srs_prj)
            add_fields_to_layer(fields=layer_meta['fields'],layer=dst_lyr)

            stream= dst_lyr.GetArrowStream()
            schema = stream.GetSchema()

            with tqdm(total=ntiles, desc=f'Downloading...') as pbar:
                for i, tiles in enumerate(chunker(all_tiles, size=16), start=1):
                    download_tasks = dict()
                    tdict = {}
                    pbar.set_postfix_str(f'downloading chunk no {i}...', refresh=True)
                    for tile in tiles :

                        tile_name = f'{tile.z}/{tile.x}/{tile.y}'
                        task = asyncio.create_task(
                            src.get_tile(z=tile.z, x=tile.x, y=tile.y),
                            name=tile_name
                        )
                        download_tasks[tile_name] = task
                        tdict[tile_name] = tile
                    if not download_tasks:continue
                    done, pending = await asyncio.wait(download_tasks.values(), timeout=1000, return_when=asyncio.ALL_COMPLETED)
                    tiles_data = dict()
                    for done_task in done:
                        try:
                            tile_name = done_task.get_name()
                            tile_data = await done_task
                            if tile_data is not None:
                                tiles_data[tile_name] = tile_data
                            else:
                                logger.debug(f'No data was downloaded for tile {tile_name} ')
                        except Exception as e:
                            logger.error(f'Failed to download tile {tile_name}. {e}')
                    if not tiles_data:continue
                    futures = dict()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                        for tile_name, tile_data in tiles_data.items():
                            kw_args = dict(data=tile_data, compression=compression,
                                           fields=layer_meta['fields'], tile=tdict[tile_name],
                                           bbox_poly=bbox_polygon)
                            futures[executor.submit(process_tile_n, **kw_args)] = tile_name

                    done, not_done = concurrent.futures.wait(futures,timeout=300, )

                    for fut in done:
                        tile_name = futures[fut]

                        exception = fut.exception()
                        if exception is not None:
                            logger.error(f'Failed to process tile {tile_name} because {exception} ')
                        arr = fut.result()
                        if arr is not None and arr.num_rows > 0:
                            logger.debug(f'Going to write {arr.num_rows} features from tile {tile_name}')

                            dst_lyr.WritePyArrow(arr)

                    #break
                pbar.set_postfix_str('Finished')


            # # Delete features outside the bbox
            # feature = dst_lyr.GetNextFeature()
            # bbox_geom = bbox2ogrgeom(bbox=bbox)
            # while feature:
            #     # Get the feature geometry
            #     geom = feature.GetGeometryRef()
            #
            #     # Check if the geometry is outside the bbox
            #     if not bbox_geom.Contains(geom):
            #         dst_lyr.DeleteFeature(feature.GetFID())  # Delete the feature if outside
            #     feature = dst_lyr.GetNextFeature()
            logger.info(f'{dst_lyr.GetFeatureCount()} feature were fetched and saved to /tmp/bldgs.fgb ')


if __name__ == '__main__':
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(100)
    logging.basicConfig()
    logger.setLevel(logging.INFO)


    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    bbox = 19.5128619671,40.9857135911,19.5464217663,41.0120783699  # ALB, Divjake
    # bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    # bbox = 15.034157,49.282809,16.02842,49.66207 # CZE

    #a =  asyncio.run(get_admin_level_bbox(iso3='ZWE'))
    #print(a)
    url = 'https://undpgeohub.blob.core.windows.net/userdata/9426cffc00b069908b2868935d1f3e90/datasets/bldgsc_20241029084831.fgb/bldgsc.pmtiles?sv=2025-01-05&ss=b&srt=o&se=2025-11-29T15%3A58%3A37Z&sp=r&sig=bQ8pXRRkNqdsJbxcIZ1S596u4ZvFwmQF3TJURt3jSP0%3D'
    validate_source()
    asyncio.run(fetch_buildings(bbox=bbox, zoom_level=None))
