
import httpx
import logging
from cbsurge.admin.osm import get_admin0_bbox, OVERPASS_API_URL
from cbsurge  import util

logger = logging.getLogger(__name__)



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