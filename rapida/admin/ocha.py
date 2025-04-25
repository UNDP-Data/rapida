"""
Fetch admin boundaries from OCHA ArcGIS server
"""
import os
import httpx
import logging
from urllib.parse import urlparse,urlencode
import pycountry
from shapely.predicates import intersects
from shapely.geometry import box, shape
import h3.api.basic_int as h3
import shapely
from rapida.admin.util import is_int
from tqdm import tqdm
from rapida.util.http_get_json import http_get_json

logger = logging.getLogger(__name__)


OCHA_COD_ARCGIS_SERVER_ROOT= 'https://codgis.itos.uga.edu/arcgis/rest/services'
ARCGIS_SERVER_ROOT = 'https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services'
ARCGIS_COD_SERVICE = 'COD_External'


def countries_for_bbox(bounding_box=None):
    """
    Retrieves the ISO 3166-1 alpha-3 country code for all  countries whose extent intersects the
    bounding box f rom ESRI ARcGIS oneline World countries layer
    :param bounding_box: tuple of numbers, xmin/west, ymin/south, xmax/east, ymax/north
    :return: tuple with string representing iso3 country codes


    """
    str_bbox = map(str, bounding_box)
    url = f'https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/World_Countries_(Generalized)/FeatureServer/0/query?where=1%3D1&outFields=*&geometry={",".join(str_bbox)}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&returnGeometry=false&outSR=4326&f=json'
    try:
        timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
        data = http_get_json(url=url, timeout=timeout)
        countries = list()
        for country in data['features']:
            iso2 = country['attributes']['ISO']
            countries.append(pycountry.countries.get(alpha_2=iso2).alpha_3)
        return tuple(countries)
    except Exception as e:
        logger.error(f'Failed to fetch  countries that intersect bbox {bounding_box}. {e}')
        raise

def fetch_ocha_countries(bounding_box = None, ):
    """
    Retrieve iso3 country codes for all countries available in OCHA COD database

    :param bounding_box:optional, if supplied only countries whose extent intersects
           the bounding box are retri
    :return:
    """
    url = os.path.join(OCHA_COD_ARCGIS_SERVER_ROOT, 'COD_External?f=pjson')
    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
    try:
        data = http_get_json(url=url, timeout=timeout)
        countries = list()
        parsed_url = urlparse(url)
        bbox_poly = box(*bounding_box) if bounding_box is not None else None
        for service in data['services']:
            service_name = service['name'].split(os.path.sep)[-1]
            try:
                service_country, service_flavour, *other = service_name.split('_')
            except ValueError:
                logger.debug(f'could not parse "{service_name}" service from {url}. Skipping.')
                continue
            service_type = service['type']
            if service_flavour == 'pcode':
                if bbox_poly is not None and service_type == 'MapServer':
                    info_url = f'{parsed_url.scheme}://{parsed_url.netloc}/{parsed_url.path}/{service_name}/{service_type}/info/iteminfo?f=json'
                    info_data = http_get_json(url=info_url, timeout=timeout)
                    country_extent = info_data['extent']
                    bottom_left, top_right = country_extent
                    west, south = bottom_left
                    east, north = top_right
                    country_bbox_poly = box(minx=west,miny=south,maxx=east,maxy=north)
                    if intersects(bbox_poly,country_bbox_poly):
                        countries.append(service_country)
                else:
                    countries.append(service_country)
        return tuple(set(countries))
    except Exception as e:
        logger.error(f'Failed to fetch available countries. {e}')
        raise


def fetch_ocha_admin_levels(iso3_country=None):
    """
    Retrieves the available admin boundaries admin levels in OCHA COD database for a given country
    :param iso3_country: str, ISO 3166-1 alpha-3 country code
    :return: dict where the keys are admin level as integer numbers and value is a tuple consisting of the
    admin layer name and its corresponding ID in the ArcGIS OCHA COD service
    Example: {0: ('Admin0', 0), 1: ('Admin1', 1), 2: ('Admin2', 2)}
    """
    url = f'{os.path.join(OCHA_COD_ARCGIS_SERVER_ROOT, ARCGIS_COD_SERVICE)}/{iso3_country}_pcode/MapServer?f=pjson'
    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
    try:
        data = http_get_json(url=url, timeout=timeout)
        rdict = {}
        for layer in data['layers']:
            layer_name = layer['name']
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


def fetch_admin(bbox=None, admin_level=None, clip=False,h3id_precision=7, ):
    """
    Retrieves administrative boundaries of a specific level/levels that intersect the area covered in the west-east and \
    south-north geographical range


    :param bbox: tuple of floats/numbers, west, south, east, north geographic coordinates
    :param admin_level:
        an integer
        a dictionary where keys are ISO3 country codes and values the corresponding admin level as int
    :param clip: bool, False, if True the admin boundaries are clipped to the bounding box
     :param: h3id_precision, default=7, the tolerance in meters (~11) use to compute the unique id as a h3 hexagon id
    :return: a python dict representing fetched administrative units in GeoJSON format

    This function uses OCHA COD geospatial services to retrieve administrative boundaries for a specific admin level
    in the area covered by a bounding box.
    Example usage:
    admin_features = fetch_admin(bbox=bbox, admin_level=2, clip=True)
    or
    admin_features = fetch_admin(bbox=bbox, admin_level={'KEN':2, 'UGA':2}, clip=True)

    if admin_features is not None:
        with open('/tmp/admin.geojson', 'wt') as out:
            out.write(json.dumps(admin_features, indent=4))

    The admin_level argument can be an integer or a dictionary.

    In case an integer is supplied, the best effort is made to retrieve admin boundaries from that specific admin
    level data layer for all countries  whose extent intersects the bounding box

    The OCA COD database is hosted on an ArcGIS server instance and is organized on a per country basis. As a result
    this function also uses a per country approach to fetch features and merge them into one layer in case the supplied
    bounding box covers several countries.

    """

    west, south, east, north = bbox
    try:
        int(admin_level)
        intersecting_countries = countries_for_bbox(bounding_box=bbox)
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


    ocha_countries = fetch_ocha_countries()


    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
    bbox_polygon = box(west, south, east, north)
    geojson = None
    with tqdm(total=len(admin_level), desc=f'Downloading ...') as pbar:
        for iso3, adm_level in admin_level.items():
            if not iso3 in ocha_countries:
                logger.info(f'OCHA database does not contain data for {iso3}')
                continue
            admin_data = fetch_ocha_admin_levels(iso3_country=iso3)
            admin_levels = tuple(admin_data)
            logger.debug(f'{iso3} contains admin data for levels {admin_levels}')
            if not adm_level in admin_levels:
                logger.info(f'OCHA COD database does not contain admin data for admin level {adm_level} in {iso3}. Available admin levels are {",".join(map(str,admin_levels))}')
                continue
            adm_layer_name, adm_layer_id = admin_data[adm_level]
            pbar.set_postfix_str(f'Fetching data for {adm_layer_name} in {iso3} ', refresh=True)
            str_bbox = f'{west}, {south}, {east}, {north}'
            qdict = dict(outFields='*',geometry=str_bbox,geometryType='esriGeometryEnvelope', inSR=4326,spatialRel='esriSpatialRelIntersects', returnGeometry='true', outSR='4326',f='GeoJSON')
            country_geojson_url = f'{OCHA_COD_ARCGIS_SERVER_ROOT}/COD_External/{iso3}_pcode/FeatureServer/{adm_layer_id}/query?{urlencode(query=qdict, doseq=True)}'
            geojson_data = http_get_json(url=country_geojson_url, timeout=timeout)
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
                        else:
                            out_props[f'admin{k1[-1]}_name'] = v

                if clip:
                    geom = geom.intersection(bbox_polygon)
                    f['geometry'] = geom.__geo_interface__
                centroid = shapely.centroid(geom)
                out_props['undp_admin_level'] = adm_level
                out_props['name_en'] = out_props['name']
                out_props['h3id'] = h3.latlng_to_cell(lat=centroid.y, lng=centroid.x, res=h3id_precision)
                out_props['iso3'] = pycountry.countries.get(alpha_2=props['ADM0_PCODE']).alpha_3
                f['properties'] = out_props
            if geojson is None:
                geojson = geojson_data
            else:
                geojson['features'] += geojson_data['features']
            pbar.update(1)

        pbar.set_postfix_str(f'Finished', refresh=True)

    return geojson





if __name__ == '__main__':
    import json
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA


    #c = fetch_admin(bbox=bbox, admin_level={'KEN':1, 'UGA':3}, clip=True)
    c = fetch_admin(bbox=bbox, admin_level=2, clip=False)
    if c is not None:
        with open('/tmp/abc.geojson', 'wt') as out:
            out.write(json.dumps(c, indent=4))