import asyncio
import gzip
import json
import math
import multiprocessing

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
from osgeo import ogr, osr, gdal
from shapely.geometry import shape, box
from shapely.io import to_wkb
from shapely import  transform, equals, normalize, area, length
import numpy as np
import itertools
import pyarrow as pa

from multiprocessing import Manager
from cbsurge.util import validate, fetch_drivers
from pyproj import Geod
ogr.UseExceptions()
osr.UseExceptions()
GMOSM_BUILDINGS = 'https://data.source.coop/vida/google-microsoft-osm-open-buildings/pmtiles/goog_msft_osm.pmtiles'


WEB_MERCATOR_TMS = m.tms.get("WebMercatorQuad")
WEB_MERCATOR_BBOX = WEB_MERCATOR_TMS.xy_bbox
WEB_MERCATOR_EXTENT = WEB_MERCATOR_BBOX.right - WEB_MERCATOR_BBOX.left
TILES_IN_CHUNK = min(multiprocessing.cpu_count(), 16)
srs_prj = osr.SpatialReference()
srs_prj.ImportFromEPSG(4326)
logger = logging.getLogger(__name__)


def validate_source():
    try:
        validate(url=GMOSM_BUILDINGS, timeout=3)
        return  True
    except Exception as e:
        msg  = f'Failed to validate buildings source {GMOSM_BUILDINGS}. Error is {e}'
        logger.error(msg)
        raise

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



async def download_mvt(src=None, tile=None, tile_name=None, tries=3 ):
    """
    Download a tile from a opened PMTiles file
    :param src: instance of opened PMTies file
    :param tile: the tile object
    :param tile_name: str
    :param tries: int, the number of times the download should be attempted
    :return:
    """
    for i in range(1, tries+1):
        logger.debug(f'Attempting ({i}) to download {tile_name} ')
        try:
            return await src.get_tile(z=tile.z, x=tile.x, y=tile.y)
        except Exception as e:
            if i == tries:
                logger.error(e)
                raise
            continue



def process_mvt(data=None, compression=None, fields=None, tile=None, bbox_poly=None, shared_dict=None, tile_buffer=80):
    """
    Process the buildings contained in the tile.
    VIDA buildings are served GZIP compressed and like most of the instances of MVT data are generated with
    a buffer that goes beyond the bounds of a tile.
    The buildings are filtered based on several  (original requested bbox) and duplicates are removed based on
    the "area_in_meters" attribute.

    TO save computational resources every building is categorized in respect to where it sits within a tile:
    inside, in buffer or at the edge. Edge buildings are not collected because they are normally split
    The inside but mainly buffer buildings can be duplicated within the bbox and this is handled through a thread safe dict
    where every building is recorded once. IN case the same building  exists the bbox  it is not collected
    unless its real area in m2 and the copy area are different.


    :param data: bytes array, the MVT binary data
    :param compression: str, the compression type
    :param fields: dict with layer's schema
    :param tile: morecantile tile instance
    :param bbox_poly: bbox to filter buildings spatially
    :param shared_dict: shared dict use between multiple threads
    :param tile_buffer: int, the number of extra cartesian coordinates added to the tile
    :return: all buildings in LATLON coords and props inside a pyarrow.table
    """
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
                    split_mask = (c==-tile_buffer) | (c==l['extent']+tile_buffer)
                    buffer_mask = ((c>-tile_buffer)&(c<0)) | ((c>l['extent'])&(c<l['extent']+tile_buffer))
                    is_split = np.any(split_mask)
                    is_tile = tile_mask.sum()==cs
                    is_buffer = buffer_mask.sum() > 0
                    if is_tile:
                        btype = 'inside'
                    if is_buffer:
                        if is_split:
                            continue
                        else:
                            btype = 'buffer'

                    tp = transform(p, transformation=partial(mvt2ll_arr, z=tile.z, x=tile.x, y=tile.y))

                    if not tp.within(bbox_poly):
                        continue

                    area_m = area(tp)
                    if not props['area_in_meters'] in shared_dict:
                        shared_dict[props['area_in_meters']] = area_m
                    else:
                        parea_m = shared_dict[props['area_in_meters']]
                        if area_m-parea_m == 0:
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

                    flds['wkb_geometry'].append(to_wkb(tp))
            r = pa.Table.from_pydict(flds)

            return r

    except Exception as e:
        logger.error(e)
        raise




def mvt2ll(x=None, y=None, tx=None, ty=None, z=None, extent=4096):
    """
    Transform vector tile int coords to geographic
    :param x: x coord, int
    :param y: y coord, int
    :param tx: tile x number
    :param ty: tile y number
    :param z: zoom level
    :param extent: int, the default size of a tile
    :return: lon/lat, tuple of floats
    """
    size = extent * 2 ** z
    x0 = extent * tx
    y0 = extent * ty
    lon = (x + x0) * 360. / size - 180
    y2 = 180 - (y + y0) * 360. / size
    lat = 360. / math.pi * math.atan(math.exp(y2 * np.pi / 180)) - 90
    return lon, lat

def mvt2ll_arr(coords_array=None, x=None, y=None, z=None, extent=4096):
    """
        Transform vector tile int coords to geographic. Array version
        :param coords_array, 2D even x and y int2 coords array

        :param x: tile x number
        :param y: tile y number
        :param z: zoom level
        :param extent: int, the default size of a tile
        :return: lon/lat, array of floats
        """
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

    """
     Create a rectangular geometry from the bounding box
    :param bbox:
    :return:
    """

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





async def fetch_buildings(bbox=None, zoom_level=None, x=None, y=None, out_path=None, url=GMOSM_BUILDINGS):
    """
    Fetch buildings from url remotye source in PMTiles format

    :param bbox:
    :param zoom_level:
    :param x:
    :param y:
    :param out_path:
    :param url:
    :return:
    """
    assert os.path.isfile(out_path), f'{out_path} has to be a file'
    out_folder, file_name = os.path.split(out_path)
    assert os.path.exists(out_folder), f'Folder {out_path} has to exist'

    if os.path.exists(out_path):
        assert os.access(out_path, os.W_OK), f'Can not write to {out_path}'


    bb = m.BoundingBox(*bbox)
    bbox_polygon = box(*bbox)
    assert WEB_MERCATOR_TMS.intersect_tms(bbox=bb) is True, (f'The supplied bounding box {bb} does not intersect the '
                                                             f'Web Mercator Quad ')

    async with Reader(url) as src:
        # PMTiles Metadata
        header = src._header
        compression = header['tile_compression'] if header['tile_compression'] is not None else None

        meta = await src.metadata()
        layer_meta = meta['vector_layers'][0]
        logger.debug(json.dumps(layer_meta, indent=2))
        minzoom, maxzoom = src.minzoom, src.maxzoom
        zoom_level = zoom_level or maxzoom-1

        all_tiles = WEB_MERCATOR_TMS.tiles(west=bb.left,south=bb.bottom, east=bb.right, north=bb.top,
                                   zooms=[zoom_level],)
        ntiles, all_tiles = generator_length(all_tiles)


        with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
            dst_lyr = dst_ds.CreateLayer('buildings', geom_type=ogr.wkbPolygon, srs=srs_prj)
            add_fields_to_layer(fields=layer_meta['fields'],layer=dst_lyr)
            bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt} tiles out of {total_fmt} [elapsed: {elapsed} remaining: {remaining}]"

            with tqdm(total=ntiles, desc=f'Downloaded', bar_format=bar_format) as pbar:
                for i, tiles in enumerate(chunker(all_tiles, size=TILES_IN_CHUNK), start=1):
                    download_tasks = dict()
                    tiles_data = dict()
                    tdict = {}
                    pbar.set_postfix_str(f'downloading chunk no {i}...', refresh=True)

                    try:
                        for tile in tiles:
                            tile_name = f'{tile.z}/{tile.x}/{tile.y}'
                            if x is not None and y is not None:
                                if tile.x != x or tile.y != y:
                                    continue
                            task = asyncio.create_task(
                                download_mvt(src=src, tile=tile, tile_name=tile_name),
                                name=tile_name
                            )
                            download_tasks[tile_name] = task
                            tdict[tile_name] = tile

                        if not download_tasks: continue
                        done, pending = await asyncio.wait(download_tasks.values(), timeout=90*len(download_tasks),
                                                           return_when=asyncio.ALL_COMPLETED)
                        failed_tasks = []
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
                                failed_tasks.append((tile_name, e))
                        # handle pending
                        for pending_task in pending:
                            try:
                                tile_name = pending_task.get_name()
                                pending_task.cancel()
                                await pending_task
                                failed_tasks.append((tile_name, asyncio.Timeout))
                            except Exception as e:
                                logger.error(f'Failed to download tile {tile_name}. {e}')
                                failed_tasks.append((tile_name, e))
                    except (asyncio.CancelledError, KeyboardInterrupt) as de:
                        logger.debug(f'Cancelling download tasks')
                        for tile_name, t in download_tasks.items():
                            if t.done():continue
                            if not t.cancelled():
                                try:
                                    t.cancel()
                                    await t
                                except asyncio.CancelledError:
                                    logger.debug(f'Download for tile {tile_name} was cancelled')
                        if de.__class__ == KeyboardInterrupt:
                            raise de
                        else:
                            break

                    if not tiles_data: continue
                    futures = dict()
                    with Manager() as manager:
                        shared_dict = manager.dict()
                        try:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                                for tile_name, tile_data in tiles_data.items():
                                    kw_args = dict(data=tile_data, compression=compression,
                                                   fields=layer_meta['fields'], tile=tdict[tile_name],
                                                   bbox_poly=bbox_polygon, shared_dict=shared_dict)
                                    futures[executor.submit(process_mvt, **kw_args)] = tile_name

                            done, not_done = concurrent.futures.wait(futures, timeout=60*len(futures), )

                            for fut in done:
                                tile_name = futures[fut]
                                exception = fut.exception()
                                if exception is not None:
                                    logger.error(f'Failed to process tile {tile_name} because {exception} ')
                                    failed_tasks.append((tile_name, exception))
                                arr = fut.result()

                                if arr is not None and arr.num_rows > 0:
                                    logger.debug(f'Going to write {arr.num_rows} features from tile {tile_name}')
                                    dst_lyr.WritePyArrow(arr)
                                    pbar.update()



                        except (concurrent.futures.CancelledError, KeyboardInterrupt) as pe:
                            logger.debug('Cancelling jobs')
                            for fut, tile_name in futures.items():
                                if fut.done():continue
                                c = fut.cancel()
                                while not c:
                                    await asyncio.sleep(.1)
                                    c= fut.cancel()
                                logger.info(f'Building extraction from tile {tile_name} was cancelled')
                            if pe.__class__ == KeyboardInterrupt:
                                raise pe
                            else:
                                break
                pbar.set_postfix_str('Finished')

            logger.info(f'{dst_lyr.GetFeatureCount()} feature were written to {out_path} ')

def remove_duplicate_buildings(src_path=None):
    """
    Remove duplicate building based on area_in_meters field
    :param src_path:
    :return:
    """
    with ogr.Open(src_path, 1) as src:
        lyr = src.GetLayer(0)
        stream = lyr.GetArrowStreamAsPyArrow()
        buffer_ftrs = dict([(b['OGC_FID'].as_py(), b['area_in_meters'].as_py()) for b in stream.GetNextRecordBatch()])
        buffer_ftrs1 = dict(zip(buffer_ftrs.values(), buffer_ftrs.keys()))
        ids_to_remove = set(buffer_ftrs.keys()).difference(buffer_ftrs1.values())
        for id_to_remove  in ids_to_remove:
            assert lyr.DeleteFeature(id_to_remove) == 0
        logger.info(f'{lyr.GetFeatureCount()} feature were saved to /tmp/bldgs.fgb ')







    # with open('/tmp/tiles.geojson', 'w') as tj:
    #     tj.write(json.dumps(dict(type='FeatureCollection', features=ftrs), indent=2))
if __name__ == '__main__':
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(100)
    logging.basicConfig()
    logger.setLevel(logging.INFO)


    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    #bbox = 19.5128619671,40.9857135911,19.5464217663,41.0120783699  # ALB, Divjake
    # bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    # bbox = 15.034157,49.282809,16.02842,49.66207 # CZE

    #a =  asyncio.run(get_admin_level_bbox(iso3='ZWE'))
    #print(a)

    validate_source()
    out_path = '/tmp/bldgs.fgb'
    asyncio.run(fetch_buildings(bbox=bbox, out_path=out_path))
