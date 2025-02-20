from osgeo import ogr
AGE_STRUCTURES_ROOT_URL = "https://hub.worldpop.org/geodata"
AZURE_BLOB_CONTAINER_NAME = "stacdata"
AZURE_FILESHARE_NAME = "cbrapida"
ARROWTYPE2OGRTYPE = {'string':ogr.OFTString, 'double':ogr.OFTReal, 'int64':ogr.OFTInteger64, 'int':ogr.OFTInteger}
DEFAULT_MASK_RESOLUTION_METERS = 100
DEFAULT_MASK_POLYGONIZATION_SMOOTHING_BUFFER = 100
GTIFF_CREATION_OPTIONS = dict(TILED='YES', COMPRESS='ZSTD', BIGTIFF='IF_SAFER', BLOCKXSIZE=256,
                                                BLOCKYSIZE=256)
POLYGONS_LAYER_NAME = 'polygons'
MASK_NAME = 'mask'