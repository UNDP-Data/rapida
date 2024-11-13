import json
import httpx
import asyncio
from osm2geojson import json2geojson
import shapely
from shapely.geometry import shape, box
import logging

OVERPASS_API_URL = 'https://overpass-api.de/api/interpreter'

logger = logging.getLogger(__name__)

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


async def get_admin0_centroid(iso3=None, overpass_url=OVERPASS_API_URL):
    """
    Retrieves from overpass the geographic centroid of the admin0 units corresponding to the
    iso3 country code
    :param iso3: ISO3 country code
    :param overpass_url:  the overpass URL
    :return: tuple of numbers representing lat/lon coordinates of the centroid
    """

    overpass_query = f"""
    [out:json];
    area["ISO3166-1:alpha3"="{iso3}"][admin_level="2"];
    rel(area)["boundary"="administrative"];
    out center;
    """



    async with httpx.AsyncClient() as client:
        response = await client.post(overpass_url, data=overpass_query)
        if response.status_code == 200:
            data = response.json()
            if 'elements' in data and len(data['elements']) > 0:
                for element in data['elements']:
                    if 'center' in element:
                        return element['center']
            return "Centroid not found"
        else:
            return f"Error: {response.status}"



async def fetch_adm_hierarchy(lat=None, lon=None, admin_level=None):
    """
    Fetches all administrative hierarchical levels from  OSM for a given point on Earth represented using geographical
    coordinates.

    :param lat: the latitude of the point
    :param lon: the longitude odf the point
    :param admin_level: the UN administrative level at which the point was generated/conceptualized
    :return: a dictionary consisting the names of the admin entitities hierachically superior to admin_level
    that exist in the OSM database

    Example for OSM level 4 corresponding ot UN admin level 2:
        lat: -0.07125779007109234 lon: 35.48194800783906 admin_props {'iso3': 'KEN', 'admin0_name': 'Kenya', 'admin1_name': 'Rift Valley'}
        lat: -0.07833364888274733 lon: 34.81953398917435 admin_props {'iso3': 'KEN', 'admin0_name': 'Kenya', 'admin1_name': 'Nyanza'}
    """
    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = f"""
                    [out:json][timeout:25];
                    is_in({lat}, {lon})->.a;
                    area.a["boundary"="administrative"];
                    out;

    """



    async with httpx.AsyncClient(timeout=100) as client:
        response = await client.post(overpass_url, data={"data": overpass_query})
        if response.status_code == 200:
            data = response.json()
            result = dict()
            if len(data['elements']) > 0:
                for element in data['elements']:
                    if element['type'] == 'area' and 'tags' in element and 'name' in element['tags'] and 'admin_level' in element['tags']:
                        tags= element['tags']
                        adm_level = int(tags['admin_level'])-2
                        if not result: result['iso3'] = tags['ISO3166-1:alpha3']
                        if adm_level >= admin_level-2:continue
                        result[f'admin{adm_level}_name'] = tags['name']
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



async def fetch_admin(west=None, south=None, east=None, north=None, admin_level=1, osm_level=3, clip=False):
    """
    Fetch admin geospatial in a LATLON bounding box from OSM Overpass API

    :param west: western coord of the bbox
    :param south: southern coord of the bbox
    :param east: eastern coord of the bbox
    :param north: northern coord of the bbox
    :param admin_level: int, (0,1,2) the UN administrative level for which to fetch data
    :param osm_level: int, None, the OSM admin level for witch to fetch data
    :param clip: boolean, False,
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
    ranging from 1 (supernational and not rendered) to 12 (noty rendered/reserved).
    In practical terms, the levels start at 2 and end at 10 (inclusive) and can be mapped to  the three UN admin levels

    using following structure {0:2, 1:(3,4,5), 2:(6,7,8)}

    This function operates in two distinct styles or modes: implicit and explicit. In the implicit style the
    highest OSM admin level that returns data is used recursively for the UN admin level specified in admin_level argument.

    In the explicit mode the specific OSM admin level specified through osm_level argument is used to fetch the data

    The returned admin layer features  the matched admin level and the name of every admin feature that was retrieved by
    the Overpass query. Additionally, all hierarchically superior admin levels and names are returned ans attributes of
    the admin units.




    """
    VALID_LEVELS = {0:2, 1:(3,4,5), 2:(6,7,8)}

    assert admin_level in VALID_LEVELS, f'Invalid admin level. Valid values are {list(VALID_LEVELS.keys())}'
    overpass_url = "https://overpass-api.de/api/interpreter"
    VALID_SUBLEVELS = VALID_LEVELS[admin_level][::-1]
    if osm_level is not None:
        assert osm_level in VALID_SUBLEVELS, f'Invalid admin osm_level. Valid values are {VALID_SUBLEVELS}'

    for i, level_value in enumerate(VALID_SUBLEVELS):
        if osm_level is not None and level_value != osm_level:continue
        logger.info(f'Fetching data for OSM level {level_value}')
        overpass_query = f"""
                               [out:json][timeout:125];
                                // Define the bounding box (replace {{bbox}} with "south,west,north,east").
                                (
                                  relation
                                    ["admin_level"="{level_value}"]  // Match admin levels 0, 1, or 2
                                    ["boundary"="administrative"] // Ensure it's an administrative boundary
                                    ["type"="boundary"]
                                    ({south}, {west}, {north}, {east});
                                );
                                
                                // Output only polygons
                                out geom;
        """
        logger.debug(overpass_query)
        async with httpx.AsyncClient() as client:
            response = await client.post(overpass_url, data={"data": overpass_query})
            if response.status_code == 200:
                data = response.json()
                if len(data['elements'])>0:
                    geojson = json2geojson(data=data)
                    nfeatures = len(geojson['features'])
                    bbox_polygon = box(west, south, east, north)
                    for f in geojson['features']:
                        props = f['properties']
                        tags = props.pop('tags')
                        feature_geom = f['geometry']
                        geom = shape(feature_geom)
                        if clip:
                            geom = geom.intersection(bbox_polygon)
                            f['geometry'] = geom.__geo_interface__
                        centroid = shapely.centroid(geom)
                        out_props = await fetch_adm_hierarchy(lat=centroid.y, lon=centroid.x, admin_level=level_value)
                        print(f'lat: {centroid.y} lon: {centroid.x} admin_props {out_props}' )
                        out_props['name'] = tags['name']
                        out_props['admin_level'] = int(tags['admin_level'])-2
                        out_props['name_en'] = tags.get('name:en', None)
                        f['properties'] = out_props

                    logger.info(f'{nfeatures} feature/s were retrieved from {overpass_url} OSM level {level_value}')


                    return geojson
                else:
                    logger.info(f'No features were  retrieved from {overpass_url} using query \n "{overpass_query}"')
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
    west, south, east, north = bbox
    c = asyncio.run(fetch_admin(west=west, south=south, east=east, north=north, admin_level=1, osm_level=None, clip=True))
    if c is not None:
        with open('/tmp/abb.geojson', 'wt') as out:
            out.write(json.dumps(c, indent=4))
        polygon = bbox_to_geojson_polygon(*bbox)
        l = {"type":"FeatureCollection", "features":[polygon]}
        with open('/tmp/bb.geojson', 'w') as out:
            out.write(json.dumps(l,indent=4))