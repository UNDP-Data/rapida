import logging

from osgeo import gdal

logger = logging.getLogger(__name__)

def fetch_drivers():
    d = dict()
    for i in range(gdal.GetDriverCount()):

        drv = gdal.GetDriver(i)
        d[drv.ShortName] = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
    return d
