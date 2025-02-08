
import httpx
import logging
from cbsurge.admin.osm import get_admin0_bbox, OVERPASS_API_URL
from cbsurge  import util
from osm2geojson import json2geojson
import geopandas
logger = logging.getLogger(__name__)

def ll2iso3(lat=None, lon=None,overpass_url=OVERPASS_API_URL ):
    query = f"""
        [out:json];
        is_in({lat},{lon})->.a;
        area.a["ISO3166-1:alpha3"];
        out body;
    """
    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
    data = util.http_post_json(url=overpass_url, data=query, timeout=timeout)
    if "elements" in data and data["elements"]:
        return data["elements"][0]["tags"].get("ISO3166-1:alpha3", None)


def bbox2admin01(bbox=None, overpass_url=OVERPASS_API_URL):
    if not bbox or len(bbox) != 4:
        raise ValueError("bbox must be a tuple (west, south, east, north)")

    west, south, east, north = bbox
    overpass_query = f"""
    [out:json][timeout:1800];
    (
      relation
        ["admin_level"="2"]
        ["boundary"="administrative"]
      ({south}, {west}, {north}, {east});
    );
    out geom;
    """

    print("Sending query to Overpass API...")
    print(overpass_query)

    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)

    try:
        response = httpx.post(overpass_url, data=overpass_query, timeout=timeout)
        print(f"Response status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if "elements" not in data or not data["elements"]:
                print("No data found for this bounding box.")
                return None  # or return an empty GeoDataFrame: gpd.GeoDataFrame()

            geojson = json2geojson(data)
            return geopandas.GeoDataFrame.from_features(geojson["features"])
        else:
            print(f"Overpass API Error: {response.text}")
            return None
    except httpx.RequestError as e:
        print(f"Request failed: {e}")
        return None


def bbox2iso31(lon_min=None, lat_min=None, lon_max=None, lat_max=None, overpass_url=OVERPASS_API_URL):


    query = f"""
    [out:json];
    (
      way({lat_min},{lon_min},{lat_max},{lon_max})["ISO3166-1:alpha3"];
      relation({lat_min},{lon_min},{lat_max},{lon_max})["ISO3166-1:alpha3"];
    );
    out tags;
    """
    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
    data = util.http_post_json(url=overpass_url,data=query,timeout=timeout)
    iso3_codes = {element['tags'].get('ISO3166-1:alpha3') for element in data.get('elements', []) if
                  'tags' in element}
    return set(iso3_codes)

def polygons2iso3(gdf=None ):
    crs = gdf.crs
    assert crs.is_geographic, f'{gdf} crs is not geographic'
    a0df = bbox2admin01(bbox=tuple(gdf.total_bounds))
    print(a0df.columns.tolist())
if __name__ == '__main__':
    from cbsurge import util
    logger = util.setup_logger(name='rapida', level=logging.INFO)
    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    lonmin, latmin, lonmax, latmax=bbox
    countries = bbox2iso31(lon_min=lonmin, lat_min=latmin, lon_max=lonmax, lat_max=latmax)

    logger.info(countries)
    for country_code in countries:
        bb = get_admin0_bbox(iso3=country_code)
        logger.info(f'Country {country_code} has bbox {bb}')