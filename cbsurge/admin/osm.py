import json
import h3.api.basic_int as h3
import httpx
import asyncio
from osm2geojson import json2geojson
import shapely
from shapely.geometry import shape, box
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)
OVERPASS_API_URL = 'https://overpass-api.de/api/interpreter'
ADMIN_LEVELS = {0: (2,), 1: (3, 4), 2: (4, 5, 6, 7, 8)}


async def get_admin0_bbox(iso3=None, overpass_url=OVERPASS_API_URL):
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
    async with httpx.AsyncClient(timeout=100) as client:
        response = await client.post(overpass_url, data=overpass_query)
        if response.status_code == 200:

            data = response.json()
            for e in data['elements']:
                bounds = e['bounds']
                return bounds['minlon'], bounds['minlat'], bounds['maxlon'], bounds['maxlat']
            else:
                raise Exception(f'No results for found for iso3 country code  {iso3}')
        else:
            return f"Error: {response.status}"


async def get_admin_centroid(iso3=None, admin_name=None, osm_admin_level=None, overpass_url=OVERPASS_API_URL):
    """
    Retrieves from overpass the geographic centroid of the admin0 units corresponding to the
    iso3 country code
    :param iso3: ISO3 country code
    :param overpass_url:  the overpass URL
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
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(overpass_url, data=overpass_query)
        if response.status_code == 200:
            data = response.json()
            if 'elements' in data and len(data['elements']) > 0:
                for element in data['elements']:
                    if 'center' in element:
                        return element['center']['lon'], element['center']['lat']
            return "Centroid not found"
        else:
            return f"Error: {response.status}"



async def fetch_adm_hierarchy(lat=None, lon=None, admin_level=None, overpass_url=OVERPASS_API_URL):
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
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(overpass_url, data={"data": overpass_query})
        if response.status_code == 200:
            data = response.json()
            result = dict()
            if len(data['elements']) > 0:
                undp_admid_levels = dict()
                for element in data['elements']:
                    if element['type'] == 'area' and 'tags' in element and 'name' in element['tags'] and 'admin_level' in element['tags']:
                        tags= element['tags']
                        adm_level = int(tags['admin_level'])
                        if adm_level > admin_level: continue
                        undp_adm_level = osmadml2undpadml(osm_level=adm_level)

                        if not result:
                            result['iso3'] = tags['ISO3166-1:alpha3']

                        if undp_adm_level in undp_admid_levels:continue
                        result[f'admin{undp_adm_level}_name'] = tags['name']
            return result

        else:
            return f"Error: {response.status_code}"

def bbox_to_geojson_polygon(west, south, east, north):
    """
    Converts a bounding box to a GeoJSON Polygon geometry.

    Parameters:
        west (float): Western longitude
        south (float): Southern latitude
        east (float): Eastern longitude
        north (float): Northern latitude

    Returns:
        dict: A GeoJSON Polygon geometry representing the bounding box.
    """
    # Define the coordinates of the bounding box as a polygon
    coordinates = [[
        [west, south],  # bottom-left corner
        [west, north],  # top-left corner
        [east, north],  # top-right corner
        [east, south],  # bottom-right corner
        [west, south]  # closing the polygon (back to bottom-left corner)
    ]]


    # Construct a GeoJSON Polygon representation of the bounding box
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        }
    }

    return geojson

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

async def fetch_admin(west=None, south=None, east=None, north=None, admin_level=None, osm_level=None,
                      clip=False,
                      h3id_precision=7, overpass_url=OVERPASS_API_URL):
    """
    Fetch admin geospatial in a LATLON bounding box from OSM Overpass API


    :param west: western coord of the bbox
    :param south: southern coord of the bbox
    :param east: eastern coord of the bbox
    :param north: northern coord of the bbox
    :param admin_level: int, (0,1,2) the UN administrative level for which to fetch data
    :param osm_level: int, None, the OSM admin level for witch to fetch data
    :param clip: boolean, False, if True geometries will be clipped to the bbox
    :param: h3id_precision, default=7, the tolerance in meters (~11) use to compute the unique id as a h3 hexagon id

    :return: a python dict representing fetched administrative units in GeoJSON format

    Example
    admin_features = asyncio.run(fetch_admin(west=west, south=south, east=east, north=north, admin_level=1, osm_level=None, clip=True))
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


    assert admin_level in ADMIN_LEVELS, f'Invalid admin level. Valid values are {list(ADMIN_LEVELS.keys())}'

    VALID_SUBLEVELS = ADMIN_LEVELS[admin_level]
    if osm_level is not None:
        assert osm_level in VALID_SUBLEVELS, f'Invalid admin osm_level. Valid values are {VALID_SUBLEVELS}'

    for i, level_value in enumerate(VALID_SUBLEVELS):
        if osm_level is not None and level_value != osm_level:continue

        overpass_query = f"""
                               [out:json][timeout:1800];
                                // Define the bounding box (replace {{bbox}} with "south,west,north,east").
                                (
                                  relation
                                    ["admin_level"="{level_value}"]  // Match admin levels 0-10
                                    ["boundary"="administrative"] // Ensure it's an administrative boundary
                                    ["type"="boundary"]
                                    ({south}, {west}, {north}, {east});
                                );
                                
                                // Output only polygons
                                out geom;
        """
        logger.debug(overpass_query)
        timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(overpass_url, data={"data": overpass_query})
            if response.status_code == 200:
                data = response.json()
                nelems = len(data['elements'])
                if nelems>0:
                    logger.info(f'Going to fetch  admin level {admin_level} boundaries from OSM level {level_value}')
                    with tqdm(total=nelems, desc=f'Downloading ...') as pbar:
                        geojson = json2geojson(data=data)
                        bbox_polygon = box(west, south, east, north)
                        for i,f in enumerate(geojson['features']):
                            props = f['properties']
                            tags = props.pop('tags')
                            pbar.set_postfix_str(f'{tags["name"]} ', refresh=True)
                            feature_geom = f['geometry']
                            geom = shape(feature_geom)
                            if clip:
                                geom = geom.intersection(bbox_polygon)
                                f['geometry'] = geom.__geo_interface__
                            centroid = shapely.centroid(geom)
                            out_props = await fetch_adm_hierarchy(lat=centroid.y, lon=centroid.x, admin_level=level_value)
                            out_props['name'] = tags['name']
                            out_props['osm_admin_level'] = level_value
                            out_props['undp_admin_level'] = admin_level
                            out_props['name_en'] = tags.get('name:en', None)
                            out_props['h3id'] = h3.latlng_to_cell(lat=centroid.y, lng=centroid.x, res=h3id_precision)
                            f['properties'] = out_props
                            pbar.update(1)
                        pbar.set_postfix_str(f'finished', refresh=True)
                        return geojson
                else:
                    #logger.info(f'No features were  retrieved from {overpass_url} using query \n "{overpass_query}"')
                    if osm_level is None:
                        logger.info(f'Moving down to OSM level {VALID_SUBLEVELS[i+1]}')
                    continue
            else:
                return f"Error: {response.status_code}"


if __name__ == '__main__':
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    bbox = 33.681335,-0.131836,35.966492,1.158979 #KEN/UGA
    #bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    bbox = 15.034157,49.282809,16.02842,49.66207 # CZE
    west, south, east, north = bbox
    c = asyncio.run(fetch_admin(west=west, south=south, east=east, north=north, admin_level=2, osm_level=7, clip=False))
    if c is not None:
        with open('/tmp/abb.geojson', 'wt') as out:
            out.write(json.dumps(c, indent=4))
        polygon = bbox_to_geojson_polygon(*bbox)
        l = {"type":"FeatureCollection", "features":[polygon]}
        with open('/tmp/bb.geojson', 'w') as out:
            out.write(json.dumps(l,indent=4))



