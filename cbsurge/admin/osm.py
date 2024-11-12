import json
import httpx
import asyncio
from osm2geojson import json2geojson
import shapely
from shapely.constructive import centroid


async def get_admin0_bbox(iso3):
    overpass_url = "http://overpass-api.de/api/interpreter"

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
                raise Exception(f'No results for found for iso3 copuntry code  {iso3}')
        else:
            return f"Error: {response.status}"


async def get_admin_level_centroid(iso3, admin_level=2):
    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = f"""
    [out:json];
    area["ISO3166-1:alpha3"="{iso3}"][admin_level="{admin_level}"];
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


async def get_admin1_unit_centroid(iso3, lat, lon):
    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = f"""
        [out:json];
            rel["ISO3166-2"~"^AF"]
            [admin_level=4]
            [type=boundary]
            [boundary=administrative];
        out center;
        """
    async with httpx.AsyncClient() as client:
        response = await client.post(overpass_url, data={"data": overpass_query})
        if response.status_code == 200:
            data = response.json()
            #print(response.text)
            if 'elements' in data and len(data['elements']) > 0:
                for element in data['elements']:
                    if element['type'] == 'relation':
                        if 'tags' in element and 'admin_level' in element['tags'] and element['tags'][
                            'admin_level'] == '1':
                            for member in element['members']:
                                if member['type'] == 'node' and 'lat' in member and 'lon' in member:
                                    return {
                                        'id': element['id'],
                                        'lat': member['lat'],
                                        'lon': member['lon'],
                                        'name': element['tags'].get('name', 'Unknown')
                                    }
            return "Admin1 unit not found"
        else:
            return f"Error: {response.status}"


async def fetch_adm_hierarchy(lat=None, lon=None, current_level=None):
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
                        if adm_level >= current_level-2:continue
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



async def fetch_admin(west=None, south=None, east=None, north=None, level=1):
    VALID_LEVELS = {0:2, 1:(3,4,5), 2:(6,7,8)}
    assert level in VALID_LEVELS, f'Invalid admin level. Valid values are {list(VALID_LEVELS.keys())}'
    overpass_url = "http://overpass-api.de/api/interpreter"
    for level_value in VALID_LEVELS[level][::-1]:
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
        print(overpass_query)
        async with httpx.AsyncClient() as client:
            response = await client.post(overpass_url, data={"data": overpass_query})
            if response.status_code == 200:
                data = response.json()
                if len(data['elements'])>0:
                    geojson = json2geojson(data=data)
                    for f in geojson['features']:
                        props = f['properties']
                        tags = props.pop('tags')

                        g = shapely.from_geojson(json.dumps(f['geometry']))
                        centroid = shapely.centroid(g)
                        out_props = await fetch_adm_hierarchy(lat=centroid.y, lon=centroid.x, current_level=level_value)

                        out_props['name'] = tags['name']
                        out_props['admin_level'] = int(tags['admin_level'])-2
                        out_props['name_en'] = tags.get('name:en', None)
                        f['properties'] = out_props
                    return geojson
                else:

                    continue
            else:
                return f"Error: {response.status_code}"




bbox = 33.681335,-0.131836,35.966492,1.158979 #KEN/UGA
bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
west, south, east, north = bbox
c = asyncio.run(fetch_admin(west=west,south=south, east=east, north=north,level=2))
with open('/tmp/abb.geojson', 'wt') as out:
    out.write(json.dumps(c, indent=4))
polygon = bbox_to_geojson_polygon(*bbox)
l = {"type":"FeatureCollection", "features":[polygon]}
with open('/tmp/bb.geojson', 'w') as out:
    out.write(json.dumps(l,indent=4))