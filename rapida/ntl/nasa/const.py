import json
from pystac_client import Client
import re
PRODUCT = 46

COLLECTIONS_STRING = \
'''
{
    "LANCEMODIS": {
        "A1": [
            "VJ146A1_NRT_2",
            "VNP46A1_NRT_1",
            "VNP46A1_NRT_2"
        ],
        "A2": [
            "VJ146A2_NRT_2",
            "VNP46A2_NRT_2"
        ],
        "A1G": [
            "VJ146A1G_NRT_2",
            "VNP46A1G_NRT_2",
            "VNP46A1G_NRT_1"
        ]
    },
    "LAADS": {
        "A1": [
            "VJ146A1_2",
            "VNP46A1_2"
        ],
        "A2": [
            "VJ146A2_2",
            "VNP46A2_2"
        ],
        "A3": [
            "VJ146A3_2",
            "VNP46A3_2",
            "VNP46A3_1"
        ],
        "A4": [
            "VJ146A4_2",
            "VNP46A4_2",
            "VNP46A4_1"
        ]
    }
}
'''
COLLECTIONS = json.loads(COLLECTIONS_STRING)

ARCHIVE = 'ARCHIVE'
OPERATIONAL = 'OPERATIONAL'
STREAM_NAMES = OPERATIONAL, ARCHIVE
CATALOGS = tuple(COLLECTIONS)
STREAMS = CATALOGS
STREAM2CATALOG = dict(zip(STREAM_NAMES, CATALOGS))
CATALOG2STREAM = {value: key for key, value in STREAM2CATALOG.items()}
CMR_STAC_ROOT = 'https://cmr.earthdata.nasa.gov/stac/'
SUB_DATASETS: dict[str:str] = {
    "A1": "DNB_At_Sensor_Radiance",
    "A1G": "DNB_At_Sensor_Radiance",
    "A2": "DNB_BRDF-Corrected_NTL",
    "A3": "AllAngle_Composite_Snow_Free",
    "A4": "NearNadir_Composite_Snow_Free"
}
PROCESSING_LEVELS = {stream_name: list(stream_data.keys()) for stream_name, stream_data in COLLECTIONS.items()}
PROCESSING_LEVEL_NAMES = {CATALOG2STREAM[stream_name]: list(stream_data.keys()) for stream_name, stream_data in COLLECTIONS.items()}

NTL_FILENAME_PATTERN = re.compile(
    r"^(?P<product>V[A-Z0-9_]+)\."
    r"A(?P<year>\d{4})(?P<doy>\d{3})\."
    r"(?:(?P<time>\d{4})\.)?"              # NEW: Optional HHMM overpass time
    r"(?P<tile>h(?P<h>\d{2})v(?P<v>\d{2}))\."
    r"(?P<version>\d{3})"
    r"(?:\.(?P<production_time>\d{13}))?"
    r"\.h5$"
)
PRODUCTS = set([item for stream in COLLECTIONS.values() for level, prod_list in stream.items() for item in prod_list])
PRODUCT_NAMES = [p.rsplit('_', 1)[0] if p.count('_') == 2 else p.split('_')[0] for p in PRODUCTS]


API_SOURCES = {
    OPERATIONAL: 'https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200', #NRT lance
    ARCHIVE: 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200' #LADDS
}
API_CONTENT = {
    OPERATIONAL: 'https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/details/allData/5200',
    ARCHIVE: 'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/5200'
}
API_PRODUCTS = {catalog: {level: sorted({prod.rsplit('_', 1)[0] for prod in products}) for level, products in levels.items()} for catalog, levels in COLLECTIONS.items()}

ROUTES = 'STAC', 'API'



def generate_collections(catalogs=CATALOGS, product_filter=PRODUCT ):
    collections = {}
    for catalog_name in catalogs:
        catalog_url = f'{CMR_STAC_ROOT}{catalog_name}'
        catalog = Client.open(catalog_url)
        for collection in catalog.get_collections():
            if f'{product_filter}' in collection.id:
                if not catalog_name in collections:
                    collections[catalog_name] = {}
                parts = collection.id.split('_')
                if len(parts) == 3:  # NRT
                    product, stream, version = parts
                    level = product[5:]
                elif len(parts) == 2:  # STD
                    product, version = parts
                    level = product[5:]
                if not level in collections[catalog_name]:
                    collections[catalog_name][level] = []
                collections[catalog_name][level].append(collection.id)
    return collections


def processing_levels(collections=COLLECTIONS):
    for stream_name, stream in collections.items():
        for processing_level, products in stream.items():
            pass

if __name__ == '__main__':

    #COLLECTIONS = generate_collections()
    #print(json.dumps(COLLECTIONS, indent=4))

    # SOURCES = tuple(COLLECTIONS)
    #
    levels = {level for stream, prods in COLLECTIONS.items() for level in stream}


    # print(PROCESSING_LEVELS)
    print(cleaned_dict)

    #print(PRODUCTS)