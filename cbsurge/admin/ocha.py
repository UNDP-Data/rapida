"""
Fetch admin boundaries from OCHA ArcGIS server
"""
import os
import httpx
import logging
from urllib.parse import urlparse,urlencode, ParseResult
import pycountry
from shapely.predicates import intersects
from shapely.geometry import box, shape
from tqdm import tqdm
import h3.api.basic_int as h3
import shapely
from cbsurge.admin.util import is_int



logger = logging.getLogger(__name__)

ARCGIS_SERVER_ROOT='https://codgis.itos.uga.edu/arcgis/rest/services'
ARCGIS_COD_SERVICE = 'COD_External'


async def fetch(url=None, timeout=None):
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            return response.json()

async def fetch_ocha_countries(url=None, bounding_box = None, ):

    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
    try:
        data = await fetch(url=url, timeout=timeout)
        countries = list()
        parsed_url = urlparse(url)
        bbox_poly = box(*bounding_box) if bounding_box is not None else None
        for service in data['services']:
            service_name = service['name'].split(os.path.sep)[-1]
            try:
                service_country, service_flavour, *other = service_name.split('_')
            except ValueError:
                logger.error(f'could not parse "{service_name}" service from {url}. SKipping.')
                continue
            service_type = service['type']
            if service_flavour == 'pcode':
                if bbox_poly is not None and service_type == 'MapServer':
                    info_url = f'{parsed_url.scheme}://{parsed_url.netloc}/{parsed_url.path}/{service_name}/{service_type}/info/iteminfo?f=json'
                    info_data = await fetch(url=info_url, timeout=timeout)
                    country_extent = info_data['extent']
                    bottom_left, top_right = country_extent
                    west, south = bottom_left
                    east, north = top_right
                    country_bbox_poly = box(minx=west,miny=south,maxx=east,maxy=north)
                    if intersects(bbox_poly,country_bbox_poly):
                        countries.append(service_country)
                else:
                    countries.append(service_country)
        return tuple(countries)
    except Exception as e:
        logger.error(f'Failed to fetch available countries. {e}')
        raise


async def countries_for_bbox(bounding_box=None):
    str_bbox = map(str, bounding_box)
    url = f'https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/World_Countries_(Generalized)/FeatureServer/0/query?where=1%3D1&outFields=*&geometry={",".join(str_bbox)}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&returnGeometry=false&outSR=4326&f=json'
    try:
        timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
        data = await fetch(url=url, timeout=timeout)
        countries = list()
        for country in data['features']:
            iso2 = country['attributes']['ISO']
            countries.append(pycountry.countries.get(alpha_2=iso2).alpha_3)
        return tuple(countries)
    except Exception as e:
        logger.error(f'Failed to fetch  countries that intersect bbox {bounding_box}. {e}')
        raise
async def fetch_admin_levels(iso3_country=None):
    url = f'{os.path.join(ARCGIS_SERVER_ROOT,ARCGIS_COD_SERVICE)}/{iso3_country}_pcode/MapServer?f=pjson'
    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
    try:
        data = await fetch(url=url, timeout=timeout)
        rdict = {}
        for layer in data['layers']:
            layer_name = layer['name']
            layer_admin_level = None
            if 'admin' in layer_name.lower() and len(layer_name) == 6:
                try:
                    layer_admin_level = int(layer_name[-1])
                    layer_id = layer['id']
                    if layer_admin_level is not None:
                        rdict[layer_admin_level] = layer_name,layer_id
                except Exception as e:
                    logger.error(f'Failed to extract admin level from layer {layer_name}')

        return rdict
    except Exception as e:
        logger.error(f'Failed to fetch available countries for {bbox}. {e}')
        raise


async def fetch_admin(west=None, south=None, east=None, north=None, admin_level=None,
                      clip=False,h3id_precision=7, ):

    try:
        int(admin_level)
        intersecting_countries = await countries_for_bbox(bounding_box=bbox)
        if not intersecting_countries:
            logger.info(f'The supplied bounding box {bbox} contains no countries')
            return
        ncountries = len(intersecting_countries)
        admin_level = dict(zip(intersecting_countries, [admin_level] * ncountries))

    except TypeError:
        assert isinstance(admin_level, dict), f'admin_level must be a dict. ex: {"KEN":1}'
        for k, v in admin_level.items():
            assert len(k) == 3, f'The admin_level dict key is not correct'
            assert pycountry.countries.get(alpha_3=k) is not  None, f'The admin_level dict key is not an iso3 country code'
            assert is_int(v), f'The value corresponding to {k} is not an integer admin level'

    countries_url = os.path.join(ARCGIS_SERVER_ROOT, 'COD_External?f=pjson')
    parsed_countries_url = urlparse(countries_url)
    ocha_countries = await fetch_ocha_countries(url=countries_url)





    # cword = 'countries' if len(intersecting_countries)> 1 else 'country'
    # logger.info(f'Bounding box {bbox} intersects {intersecting_countries} {cword}')
    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
    bbox_polygon = box(west, south, east, north)
    geojson = None
    for iso3, adm_level in admin_level.items():
        if not iso3 in ocha_countries:
            logger.info(f'OCHA database does not contain data for {iso3}')
            continue
        admin_data = await fetch_admin_levels(iso3_country=iso3)
        admin_levels = tuple(admin_data)
        logger.debug(f'{iso3} contains admin data for levels {admin_levels}')
        if not adm_level in admin_levels:
            logger.info(f'OCHA COD database does not contain admin data for {admin_level} in {iso3}. Available admin levels are {admin_levels}')
            continue
        adm_layer_name, adm_layer_id = admin_data[adm_level]
        str_bbox = f'{west}, {south}, {east}, {north}'
        qdict = dict(outFields='*',geometry=str_bbox,geometryType='esriGeometryEnvelope', inSR=4326,spatialRel='esriSpatialRelIntersects', returnGeometry='true', outSR='4326',f='GeoJSON')
        country_url_res = ParseResult(scheme=parsed_countries_url.scheme,
                                      netloc=parsed_countries_url.netloc,
                                      path=f'{parsed_countries_url.path}/{iso3}_pcode/FeatureServer/{adm_layer_id}/query',
                                      params=None,
                                      query=urlencode(query=qdict,doseq=True),
                                      fragment=None)
        country_geojson_url = country_url_res.geturl()

        geojson_data = await fetch(url=country_geojson_url, timeout=timeout)
        nfeatures = len(geojson_data['features'])

        with tqdm(total=nfeatures, desc=f'Augmenting admin level {admin_level} features from {iso3} ...') as pbar:
            for i, f in enumerate(geojson_data['features']):
                props = f['properties']
                feature_geom = f['geometry']
                geom = shape(feature_geom)
                out_props = {}
                for k, v in props.items():
                    if not '_' in k:continue
                    k1, k2, *rest = k.split('_')
                    if k1.lower().startswith('adm') and k1[-1].isnumeric() and len(k1)==4 and k2 == 'EN':
                        if int(adm_level) == int(k1[-1]):
                            out_props['name'] = v
                            pbar.set_postfix_str(v , refresh=True)
                        else:
                            out_props[f'admin{k1[-1]}_name'] = v

                if clip:
                    geom = geom.intersection(bbox_polygon)
                    f['geometry'] = geom.__geo_interface__
                centroid = shapely.centroid(geom)
                out_props['undp_admin_level'] = adm_level
                out_props['name_en'] = out_props['name']
                out_props['h3id'] = h3.latlng_to_cell(lat=centroid.y, lng=centroid.x, res=h3id_precision)
                f['properties'] = out_props
                pbar.update(1)
            pbar.set_postfix_str(f'finished', refresh=True)
        if geojson is None:
            geojson = geojson_data
        else:
            geojson['features'] += geojson_data['features']


    return geojson





if __name__ == '__main__':
    import asyncio
    import json
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    west, south, east, north = bbox

    c = asyncio.run(fetch_admin(west=west, south=south, east=east, north=north, admin_level={'KEN':1, 'UGA':3}, clip=False))
    if c is not None:
        with open('/tmp/abc.geojson', 'wt') as out:
            out.write(json.dumps(c, indent=4))