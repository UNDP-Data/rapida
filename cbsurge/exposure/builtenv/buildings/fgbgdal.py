import os.path
import tempfile
import time
from osgeo_utils import ogrmerge
from cbsurge import util as u
from osgeo import ogr, gdal
import httpx
from cbsurge.admin.osm import OVERPASS_API_URL
from cbsurge.util import validate
import logging



gdal.UseExceptions()
ogr.UseExceptions()
logger = logging.getLogger(__name__)

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

def download(bbox=None, out_path =None):
        """
        Download building from VIDA source using GDAL
        :param bbox: iterable of floats, xmin, ymin, xmax,ymax
        :param out_path: str, full path wheer the buildings layer will be written
        :return:
        """
        u.validate_path(out_path)
        options = gdal.VectorTranslateOptions(
            format='FlatGeobuf',
            spatFilter=bbox,
            layerName='buildings'
        )

        with tempfile.TemporaryDirectory(delete=True) as tmp_dir:
            files_to_merge = []
            for country in get_countries_for_bbox_osm(bbox=bbox):
                remote_country_fgb_url = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={country}/{country}.fgb'
                local_country_fgb_url = os.path.join(tmp_dir,f'buildings_{country}.fgb' )
                ds = gdal.VectorTranslate(local_country_fgb_url,remote_country_fgb_url,options=options)
                ds = None
                files_to_merge.append(local_country_fgb_url)
                assert os.path.exists(local_country_fgb_url)
            logger.debug(f'merging {",".join(files_to_merge)} into {out_path}')
            ogrmerge.ogrmerge(src_datasets=files_to_merge, dst_filename=out_path,single_layer=True, progress_arg=ogr.TermProgress_nocb, overwrite_ds=True)
        with ogr.Open(out_path) as src:
            l = src.GetLayer(0)
            logger.info(f'{l.GetFeatureCount()} buildings were downloaded to {out_path}')




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
    bbox = 19.350384,41.206737,20.059003,41.571459 # ALB, TIRANA
    bbox = 19.726666,39.312705,20.627545,39.869353, # ALB/GRC

    #a =  asyncio.run(get_admin_level_bbox(iso3='ZWE'))
    #print(a)
    url = 'https://undpgeohub.blob.core.windows.net/userdata/9426cffc00b069908b2868935d1f3e90/datasets/bldgsc_20241029084831.fgb/bldgsc.pmtiles?sv=2025-01-05&ss=b&srt=o&se=2025-11-29T15%3A58%3A37Z&sp=r&sig=bQ8pXRRkNqdsJbxcIZ1S596u4ZvFwmQF3TJURt3jSP0%3D'
    #validate_source()
    out_path = '/tmp/bldgs.fgb'
    cntry = get_countries_for_bbox_osm(bbox=bbox)
    start = time.time()
    #asyncio.run(download(bbox=bbox))

    download(bbox=bbox, out_path=out_path)

    end = time.time()
    print((end-start))