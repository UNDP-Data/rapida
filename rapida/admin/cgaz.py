import os

import h3.api.basic_int as h3
import logging
from osgeo import gdal, ogr
from rich.progress import Progress
from rapida.util.geo import gdal_callback
from shapely.geometry import box


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
        task = progress.add_task("[cyan]Fetching admin boundaries...", total=2)

        translate_task = progress.add_task(description="Downloading ADMIN data", total=None)

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


        with gdal.config_options(options=configs):
            ds = gdal.VectorTranslate(destination_path, url, options=options)
            if ds is None:
                logger.error(f"GDAL VectorTranslate failed for URL: {url} with bbox: {bbox}")
                progress.update(task, completed=5, description="[red]GDAL VectorTranslate failed")
                return None
            ds = None
            if not clip:
                l = ds.GetLayerByName(dst_layer_name or f'admin{admin_level}')
                layer_extent = l.GetExtent()

                layer_bbox_geom = box(minx=layer_extent[0], miny=layer_extent[2], maxx=layer_extent[1], maxy=layer_extent[3])
                bbox_geometry = box(*bbox)


                ratio = layer_bbox_geom.area / bbox_geometry.area
                if ratio < 0.9 or ratio > 1.1:
                    destination_path1 = f"{destination_path}.fgb"
                    ds = None
                    ds1 = gdal.VectorTranslate(destNameOrDestDS=destination_path1, srcDS=destination_path, clipSrc=bbox, geometryType="MultiPolygon",)
                    ds1 = None
                    os.remove(destination_path)
                    os.rename(destination_path1, destination_path)

            ds = None
        progress.remove_task(translate_task)

        progress.update(task, advance=1, description="[green]Downloaded admin data", refresh=True)
        with gdal.OpenEx(destination_path, gdal.OF_VECTOR|gdal.OF_UPDATE) as ds:
            layer = ds.GetLayerByName(dst_layer_name or f'admin{admin_level}')
            iso3_field_index = layer.GetLayerDefn().GetFieldIndex("iso3")

            if iso3_field_index >=0:
                countries = set([f.GetField("iso3") for f in layer])
                if len(countries) == 0:
                    raise Exception(f"No countries were found in {bbox}")
            else:
                raise Exception(f'{url} does not contain iso3 field!')

            h3id_field_index = layer.GetLayerDefn().GetFieldIndex("h3id")
            if h3id_field_index < 0:
                h3id_field = ogr.FieldDefn("h3id", ogr.OFTInteger64)
                layer.CreateField(h3id_field)
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    centroid = geom.Centroid()
                    h3id = h3.latlng_to_cell(lat=centroid.GetY(), lng=centroid.GetX(),
                                      res=7)
                    feature.SetField("h3id", h3id)
                    layer.SetFeature(feature)
                ds.FlushCache()
        progress.update(task, advance=1, description="[green]Download Completed", refresh=True)
        return None

