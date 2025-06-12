import json
import httpx
import h3.api.basic_int as h3
from rapida.util.http_post_json import http_post_json
from osm2geojson import json2geojson
import shapely
from shapely.geometry import shape, box
import logging
from tqdm import tqdm
from rapida.admin.util import bbox_to_geojson_polygon

logger = logging.getLogger(__name__)
OVERPASS_API_URL = 'https://overpass-api.de/api/interpreter'
ADMIN_LEVELS = {0: (2,), 1: (3, 4), 2: (4, 5, 6, 7, 8)}


def get_admin0_bbox(iso3=None, overpass_url=OVERPASS_API_URL):
    """
    Retrieves from overpass the geographic bounding box of the admin0 units corresponding to the
    iso3 country code
    :param iso3: ISO3 country code
    :param overpass_url: the overpass URL,
    :return: a tuple of numbers representing the geographical bbox as west, south, east, north
    """

    overpass_query = \
        f"""
        [out:json];
        area["ISO3166-1:alpha3"="{iso3}"][admin_level="2"];
        rel(pivot)["boundary"="administrative"];
        out bb;

    """
    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
    data = http_post_json(url=overpass_url, query=overpass_query, timeout=timeout)
    for e in data['elements']:
        bounds = e['bounds']
        return bounds['minlon'], bounds['minlat'], bounds['maxlon'], bounds['maxlat']
    else:
        raise Exception(f'No results for found for iso3 country code  {iso3}')


def get_admin_centroid(iso3=None, admin_name=None, osm_admin_level=None, overpass_url=OVERPASS_API_URL):
    """
    Retrieves from overpass the geographic centroid of the admin0 units corresponding to the
    iso3 country code
    :param iso3: ISO3 country code
    :param overpass_url:  the overpass URL
    :param admin_name
    :param osm_admin_level
    :return: tuple of numbers representing lat/lon coordinates of the centroid
    """

    overpass_query = f"""
                [out:json][timeout:360];
               // Define the area using the ISO3 country code
                area["ISO3166-1:alpha3"="{iso3}"]->.country_area;

                relation
                  ["name"="{admin_name}"]
                  ["admin_level"="{osm_admin_level}"]
                  (area.country_area);
                out center;
    """

    timeout = httpx.Timeout(connect=10, read=1800, write=1800,pool=1000)
    try:
        data = http_post_json(url=overpass_url, query=overpass_query, timeout=timeout)
        for element in data['elements']:
            return element['center']['lon'], element['center']['lat']
    except Exception as e:
        raise Exception(f'Could not fetch centroid for {iso3}:{admin_name}. {e}')





def fetch_adm_hierarchy(lat=None, lon=None, admin_level=None, overpass_url=OVERPASS_API_URL):
    """
    Fetches all administrative hierarchical levels from  OSM for a given point on Earth represented using geographical
    coordinates.

    :param overpass_url: str
    :param lat: the latitude of the point
    :param lon: the longitude odf the point
    :param admin_level: the UN administrative level at which the point was generated/conceptualized
    :param compute_undp_admin_id: bool, False, flag to indicate if UNDOP admid will be computed
    :return: a dictionary consisting the names of the admin entities hierarchically superior to admin_level
    that exist in the OSM database and the iso3 country code

    Example for OSM level 4 corresponding ot UN admin level 2:
        lat: -0.07125779007109234 lon: 35.48194800783906 admin_props {'iso3': 'KEN', 'admin0_name': 'Kenya', 'admin1_name': 'Rift Valley'}
        lat: -0.07833364888274733 lon: 34.81953398917435 admin_props {'iso3': 'KEN', 'admin0_name': 'Kenya', 'admin1_name': 'Nyanza'}
    """


    overpass_query = f"""
                    [out:json][timeout:360];
                    is_in({lat}, {lon})->.a;
                    area.a["boundary"="administrative"];
                    out;

    """


    timeout = httpx.Timeout(connect=10, read=1800, write=1800,pool=1000)
    try:
        data = http_post_json(url=overpass_url, query=overpass_query, timeout=timeout)
        result = dict()
        if len(data['elements']) > 0:
            undp_admid_levels = dict()
            for element in data['elements']:
                if element['type'] == 'area' and 'tags' in element and 'name' in element['tags'] and 'admin_level' in \
                        element['tags']:
                    tags = element['tags']
                    adm_level = int(tags['admin_level'])
                    if adm_level > admin_level: continue
                    undp_adm_level = osmadml2undpadml(osm_level=adm_level)
                    if not result:
                        result['iso3'] = tags.get('ISO3166-1:alpha3', None)
                    if undp_adm_level in undp_admid_levels: continue
                    result[f'admin{undp_adm_level}_name'] = tags['name']
        return result
    except Exception as e:
        logger.error(f'Failed to fetch admin hierarchy for {lat}:{lon} with query {overpass_query}')
        raise e


def osmadml2undpadml(osm_level=None):
    """
    Convert OSM admin level to UNDP admin level
    :param osm_level: int, OSM admin levels from 2-10
    :return: UNDO adm level equivalent


    """
    for k, v in ADMIN_LEVELS.items():
        if osm_level in v:
            return k
def undpadml2osmadml(undp_level=None):
    return ADMIN_LEVELS[undp_level][-1]

def fetch_admin( bbox=None, admin_level=None, osm_level=None,
                       clip=False, h3id_precision=7, overpass_url=OVERPASS_API_URL
                    ):
    """
    Fetch admin geospatial in a LATLON bounding box from OSM Overpass API


    :param bbox: tuple of floats/numbers, west, south, east, north geographic coordinates
    :param admin_level: int, (0,1,2) the UN administrative level for which to fetch data
    :param osm_level: int, None, the OSM admin level for witch to fetch data
    :param clip: boolean, False, if True geometries will be clipped to the bbox
    :param: h3id_precision, default=7, the tolerance in meters (~11) use to compute the unique id as a h3 hexagon id

    :return: a python dict representing fetched administrative units in GeoJSON format

    Example
    admin_features = fetch_admin(bbox=bbox, admin_level=1, osm_level=None, clip=True)
    if admin_features is not None:
        with open('/tmp/admin.geojson', 'wt') as out:
            out.write(json.dumps(admin_features, indent=4))

    The issue of global level administrative units is a long term problem. The largest issue by far is represented by
    disputed areas (wars, conflicts, etc.) and the shear heterogeneity of the countries in terms of size.
    The simple fact is that every country is unique and features one or more admin levels where level 0 corresponds to
    countries, level 1 to the next hierarchical subdivisions (countries, regions, provinces) and level 2 corresponds to
    next hierarchical subdivisions after level 1 (regions, provinces, districts, etc.)

    While there is a general understanding and agreement on these three levels there is no agreement on what each level
    represents/should represent.

    OSM is a great source of open data and here the Overpass "https://overpass-api.de/api/interpreter" API is used to
    fetch the geospatial representation of administrative divisions. OSM has its own levels described at
    https://wiki.openstreetmap.org/wiki/Tag:boundary%3Dadministrative#Super-national_administrations
    ranging from 1 (supra national and not rendered) to 12 (noty rendered/reserved).
    In practical terms, the levels start at 2 and end at 10 (inclusive) and can be mapped to  the three UN admin levels

    using following structure {0:2, 1:(3,4), 2:(4,5,6,7,8)}

    This function operates in two distinct styles or modes: implicit and explicit. In the implicit style the
    highest OSM admin level that returns data is used recursively for the UN admin level specified in admin_level argument.

    In the explicit mode the specific OSM admin level specified through osm_level argument is used to fetch the data

    The returned admin layer features  the matched admin level and the name of every admin feature that was retrieved by
    the Overpass query. Additionally, all hierarchically superior admin levels and names are returned and attributes of
    the admin units.

    H3
    Resolution	Average edge
                length (meters)
    0	        11,000
    1	        4,190
    2	        1,560
    3	        597
    4	        224
    5	        84.4
    6	        31.6
    7	        11.2
    8	        4.22
    9	        1.57
    10	        0.592
    11	        0.222


    """
    if type(admin_level) is str:
        admin_level = int(admin_level)
    if type(osm_level) is str:
        osm_level = int(osm_level)

    west, south, east, north = bbox
    assert admin_level in ADMIN_LEVELS, f'Invalid admin level. Valid values are {list(ADMIN_LEVELS.keys())}'

    VALID_SUBLEVELS = ADMIN_LEVELS[admin_level]
    if osm_level is not None:
        assert osm_level in VALID_SUBLEVELS, f'Invalid admin osm_level. Valid values are {VALID_SUBLEVELS}'

    for i, level_value in enumerate(VALID_SUBLEVELS):
        if osm_level is not None and level_value != osm_level:continue
        overpass_query = f"""
                            [out:json][timeout:1800];
                            (
                              relation
                                ["admin_level"="{level_value}"]  // Match admin levels 0-10
                                ["boundary"="administrative"] // Ensure it's an administrative boundary
                                ["type"="boundary"]
                                ({south}, {west}, {north}, {east});
                            );
                            out geom;"""

        logger.debug(overpass_query)
        timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
        try:
            data = http_post_json(url=overpass_url, query=overpass_query, timeout=timeout)
            nelems = len(data['elements'])
            if nelems>0:
                logger.info(f'Going to fetch  admin level {admin_level} boundaries from OSM level {level_value}')
                with tqdm(total=nelems, desc=f'Downloading ...') as pbar:
                    geojson = json2geojson(data=data)
                    bbox_polygon = box(west, south, east, north)
                    for i,f in enumerate(geojson['features']):
                        props = f['properties']
                        tags = props.pop('tags')
                        pbar.set_postfix_str(f'{tags.get("name", None)} ', refresh=True)
                        feature_geom = f['geometry']
                        geom = shape(feature_geom)
                        if clip:
                            geom = geom.intersection(bbox_polygon)
                            f['geometry'] = geom.__geo_interface__
                        centroid = shapely.centroid(geom)
                        out_props = fetch_adm_hierarchy(lat=centroid.y, lon=centroid.x, admin_level=level_value)
                        out_props['name'] = tags.get('name', None)
                        out_props['osm_admin_level'] = level_value
                        out_props['undp_admin_level'] = admin_level
                        out_props['name_en'] = tags.get('name:en', None)
                        out_props['h3id'] = h3.latlng_to_cell(lat=centroid.y, lng=centroid.x, res=h3id_precision)
                        f['properties'] = out_props
                        pbar.update(1)
                    pbar.set_postfix_str(f'finished', refresh=True)
                    return geojson
            else:
                logger.info(f'No features were  retrieved from {overpass_url} using query \n "{overpass_query}"')
                logger.info(f'Try changing OSM level or omitting it so eventually an OSM level is found!')
                if osm_level is None:
                    logger.info(f'Moving down to OSM level {VALID_SUBLEVELS[i+1]}')
                continue
        except Exception as e:
            raise e
            logger.error(f'Can not fetch admin data from {overpass_url} at this time. {e}')
            logger.error(overpass_query)








if __name__ == '__main__':
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    # bbox = 33.681335,-0.131836,35.966492,1.158979 #KEN/UGA
    #bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    #bbox = 15.034157,49.282809,16.02842,49.66207 # CZE
    bbox = 65.742188,5.441022,90.351563,35.029996

    # c = fetch_admin(bbox=bbox, admin_level=2, osm_level=4, clip=False)
    # if c is not None:
    #     with open('/tmp/abb.geojson', 'wt') as out:
    #         out.write(json.dumps(c, indent=4))
    polygon = bbox_to_geojson_polygon(*bbox)
    l = {"type":"FeatureCollection", "features":[polygon]}
    with open('/tmp/bb.geojson', 'w') as out:
        out.write(json.dumps(l,indent=4))



