import re
PRODUCTS_RE = {
    'CM': re.compile(r'^(?P<product>[\w-]+)_(?P<version>v\d+r\d+)_(?P<platform>\w+)_s(?P<start>\d+)_e(?P<end>\d+)_c(?P<creation>\d+)\.(?P<ext>\w+)$'),
    'GEO': re.compile(r'^(?P<product>[^_]+)_(?P<platform>[^_]+)_d(?P<date>\d{8})_t(?P<start>\d+)_e(?P<end>\d+)_b(?P<orbit>\d+)_c(?P<creation>\d+)_(?P<facility>[^_]+)_(?P<env>[^_]+)\.(?P<ext>\w+)$'),
    'SDR': re.compile(r'^(?P<product>[^_]+)_(?P<platform>[^_]+)_d(?P<date>\d{8})_t(?P<start>\d+)_e(?P<end>\d+)_b(?P<orbit>\d+)_c(?P<creation>\d+)_(?P<facility>[^_]+)_(?P<env>[^_]+)\.(?P<ext>\w+)$')
}


PRODUCTS={
    'SDR':"VIIRS-DNB-SDR",
    'GEO':"VIIRS-DNB-GEO",
    'CM':"VIIRS-JRR-CloudMask"
}
PRODUCT2NAME = dict((v, k) for k, v in PRODUCTS.items())


PRODUCT_NAMES = tuple(PRODUCTS)
SOURCES = dict(aws='aws', gcp='gcp')
SOURCE_NAMES=tuple(SOURCES)
# Define the base URLs for the three VIIRS satellites
VIIRS_URLS = {
    "SNPP": {
        "aws": "s3://noaa-nesdis-snpp-pds/",
        "gcp": "gs://gcp-noaa-nesdis-snpp/"
    },
    "N20": {
        "aws": "s3://noaa-nesdis-n20-pds/",
        "gcp": "gs://gcp-noaa-nesdis-n20/"
    },
    "N21": {
        "aws": "s3://noaa-nesdis-n21-pds/",
        "gcp": "gs://gcp-noaa-nesdis-n21/"
    }
}
PUBLIC_CONFIG = {"skip_signature": "true"}