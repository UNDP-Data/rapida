import os.path

from osgeo import ogr

import httpx
from cbsurge.admin.osm import OVERPASS_API_URL
from cbsurge.util import validate
import logging
logger = logging.getLogger(__name__)
import tempfile
GMOSM_BUILDINGS_ROOT = 'https://data.source.coop/vida/google-microsoft-osm-open-buildings/flatgeobuf/by_country'
ogr.UseExceptions()
def validate_source():
    try:
        validate(url=GMOSM_BUILDINGS_ROOT)
    except Exception as e:
        logger.error(f'Failed to validate source {GMOSM_BUILDINGS_ROOT}. Error is {e}')


def get_countries_for_bbox_osm(bbox=None, overpass_url=OVERPASS_API_URL):
    """
    Retrieves from overpass the geographic bounding box of the admin0 units corresponding to the
    iso3 country code
    :param iso3: ISO3 country code
    :param overpass_url: the overpass URL,
    :return: a tuple of numbers representing the geographical bbox as west, south, east, north
    """
    west, south, east, north = bbox
    overpass_query = \
        f"""
        [out:json][timeout:1800];
                                // Define the bounding box (replace {{bbox}} with "south,west,north,east").
                                (
                                  relation["admin_level"="2"]["boundary"="administrative"]({south}, {west}, {north}, {east});  // Match admin levels 0-10 // Ensure it's an administrative boundary
                                    
                                );
                                
        /*added by auto repair*/
        (._;>;);
        /*end of auto repair*/
        out body;

    
    """
    timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(overpass_url, data={"data": overpass_query})
            response.raise_for_status()
            data = response.json()
            countries = [e['tags']['ISO3166-1:alpha3'] for e in data['elements'] if e['type'] == 'relation' and 'tags' in e]
            return tuple(countries)

    except Exception as e:
        logger.error(f'Failed to fetch available countries from OSM. {e}')
        raise

async def fetch_buildings(bbox=None):

    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource('/tmp/bldgs.fgb') as dst_ds:
        for country in get_countries_for_bbox_osm(bbox=bbox):
            country_fgb_url = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={country}/{country}.fgb'
            c = f'ogr2ogr -spat {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} /tmp/bldgs_{country}_rect.fgb {country_fgb_url}'
            logger.info(c)
            with ogr.Open(country_fgb_url) as src_ds:
                src_lyr = src_ds.GetLayer(0)
                print(src_lyr.GetFeatureCount())

                src_lyr.SetSpatialFilterRect(*bbox)
                src_lyr.ResetReading()
                print(src_lyr.GetFeatureCount())

                if dst_ds.GetLayerCount() == 0:
                    dst_lyr = dst_ds.CreateLayer(f'bldgs', geom_type=ogr.wkbPolygon, srs=src_lyr.GetSpatialRef())
                    stream = dst_lyr.GetArrowStream(["MAX_FEATURES_IN_BATCH=300"])
                    schema = stream.GetSchema()
                    for i in range(schema.GetChildrenCount()):
                        if schema.GetChild(i).GetName() != src_lyr.GetGeometryColumn():
                            dst_lyr.CreateFieldFromArrowSchema(schema.GetChild(i))

                #src_lyr.ResetReading()



                while True:
                    array = stream.GetNextRecordBatch()
                    print(array)
                    if array is None:
                        break
                    print(array.num_rows)
                    assert dst_lyr.WriteArrowBatch(schema, array) == ogr.OGRERR_NONE


if __name__ == '__main__':
    import asyncio
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(100)
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    nf = 5829
    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    #bbox = 19.5128619671,40.9857135911,19.5464217663,41.0120783699  # ALB, Divjake
    # bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    # bbox = 15.034157,49.282809,16.02842,49.66207 # CZE

    #a =  asyncio.run(get_admin_level_bbox(iso3='ZWE'))
    #print(a)
    url = 'https://undpgeohub.blob.core.windows.net/userdata/9426cffc00b069908b2868935d1f3e90/datasets/bldgsc_20241029084831.fgb/bldgsc.pmtiles?sv=2025-01-05&ss=b&srt=o&se=2025-11-29T15%3A58%3A37Z&sp=r&sig=bQ8pXRRkNqdsJbxcIZ1S596u4ZvFwmQF3TJURt3jSP0%3D'
    #validate_source()
    get_countries_for_bbox_osm(bbox=bbox)
    asyncio.run(fetch_buildings(bbox=bbox,))
