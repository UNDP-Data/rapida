import os.path

import h3.api.basic_int as h3
import logging
from osgeo import gdal, ogr
from rich.progress import Progress


gdal.UseExceptions()
CGAZ_GEOBOUNDARIES_ROOT = "https://www.geoboundaries.org/api/current/gbHumanitarian"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def fetch_admin(bbox=None, admin_level=None, clip=False, destination_path=None, dst_layer_name=None):
    """
    Fetch ADMIN from CGAZ
    Parameters
    ----------
    bbox: The bounding box to fetch administrative units for.
    admin_level: The administrative level to fetch. Should be an integer.
    clip: bool, False, if True, the admin boundaries are clipped to the bounding box.
    dst_layer_name: The layer name of the destination
    destination_path: The destination path of where to save the dataset

    Returns
    -------

    """
    with Progress() as progress:
        task = progress.add_task("[cyan]Fetching admin boundaries...", total=5)

        progress.update(task, advance=1, description="[green]Checking intersecting countries")

        progress.update(task, advance=1, description="[green]Setting GDAL options")
        options = gdal.VectorTranslateOptions(
            layerName=dst_layer_name or f'admin{admin_level}',
            makeValid=True,
            spatFilter=bbox,
            clipSrc=bbox if clip else None,
            geometryType="MultiPolygon",
        )

        url = f"/vsicurl/https://undpngddlsgeohubdev01.blob.core.windows.net/admin/cgaz/geoBoundariesCGAZ_ADM{admin_level}.fgb"

        configs = {
            'GDAL_DISABLE_READDIR_ON_OPEN': 'TRUE',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.fgb',
            'OGR2OGR_USE_ARROW_API': 'YES'
        }

        progress.update(task, advance=1, description="[green]Downloading and translating with GDAL")
        with gdal.config_options(options=configs):
            ds = gdal.VectorTranslate(destination_path, url, options=options)
            if ds is None:
                logger.error(f"GDAL VectorTranslate failed for URL: {url} with bbox: {bbox}")
                progress.update(task, completed=5, description="[red]GDAL VectorTranslate failed")
                return None
            ds = None
        with gdal.OpenEx(destination_path, gdal.OF_VECTOR|gdal.OF_UPDATE) as ds:

            layer = ds.GetLayerByName(dst_layer_name or f'admin{admin_level}')
            countries = set([f.GetField("iso_3") for f in layer])
            if len(countries) == 0:
                raise Exception(f"No countries were found in {bbox}")

            iso3_field_index = layer.GetLayerDefn().GetFieldIndex("iso_3")
            if iso3_field_index >= 0:
                new_fdefn = ogr.FieldDefn("iso3", ogr.OFTString)
                layer.CreateField(new_fdefn)
            i = layer.GetLayerDefn().GetFieldIndex("shapeID")
            if i >= 0:
                layer.DeleteField(i)
                h3id_field = ogr.FieldDefn("h3id", ogr.OFTInteger64)
                layer.CreateField(h3id_field)
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    centroid = geom.Centroid()
                    h3id = h3.latlng_to_cell(lat=centroid.GetY(), lng=centroid.GetX(),
                                      res=7)
                    feature.SetField("h3id", h3id)
                    iso3_country_code = feature.GetField("iso_3")
                    feature.SetField("iso3", iso3_country_code)
                    layer.SetFeature(feature)
                iso3_field_index = layer.GetLayerDefn().GetFieldIndex("iso_3")
                layer.DeleteField(iso3_field_index)
            ds.FlushCache()
            ds.SyncToDisk()
        progress.update(task, completed=5, description="[green]Download Completed")
        return None

