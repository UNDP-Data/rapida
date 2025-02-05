from geopandas import read_file
import logging

from pycountry import countries

logger = logging.getLogger(__name__)
ADMIN0_SOURCE='https://undpgeohub.blob.core.windows.net/rapida/admin/adm0_polygons.fgb'




def bbox2iso3(lon_min=None, lat_min=None, lon_max=None, lat_max=None, admin0_src=ADMIN0_SOURCE):
    """
    Compute the countries intersecting  a geographic bounding box in
    https://undpgeohub.blob.core.windows.net/rapida/admin/adm0_polygons.fgb dataset
    :param lon_min: number, left  bound
    :param lat_min: number, bottom bound
    :param lon_max: number, right_bound
    :param lat_max: number, top bound
    :param admin0_src, str, default: https://undpgeohub.blob.core.windows.net/rapida/admin/adm0_polygons.fgb
    :return: iterable of iso3 country codes
    """
    a0_src = f'/vsicurl/{admin0_src}'
    df = read_file(filename=a0_src, bbox=(lon_min, lat_min, lon_max, lat_max),columns=['iso_3'], ignore_geometry=True, engine='pyogrio')
    return set(df['iso_3'].tolist())



if __name__ == '__main__':
    from cbsurge import util
    logger = util.setup_logger(name='rapida', level=logging.INFO)
    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    lonmin, latmin, lonmax, latmax=bbox
    countries = bbox2iso3(lon_min=lonmin, lat_min=latmin, lon_max=lonmax, lat_max=latmax)
    logger.info(countries)