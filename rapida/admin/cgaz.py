import tempfile
import httpx
import logging
import os
import geopandas as gpd

import pycountry
from osgeo import gdal
from rich.progress import Progress
from shapely.geometry import box

from rapida.util.http_get_json import http_get_json

gdal.UseExceptions()
CGAZ_GEOBOUNDARIES_ROOT = "https://www.geoboundaries.org/api/current/gbHumanitarian"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def countries_for_bbox(bounding_box=None):
    str_bbox = map(str, bounding_box)
    url = f'https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/World_Countries_(Generalized)/FeatureServer/0/query?where=1%3D1&outFields=*&geometry={",".join(str_bbox)}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&returnGeometry=false&outSR=4326&f=json'
    try:
        timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
        data = http_get_json(url=url, timeout=timeout)
        countries = [pycountry.countries.get(alpha_2=country['attributes']['ISO']).alpha_3 for country in data['features']]
        return tuple(countries)
    except Exception as e:
        logger.error(f'Failed to fetch countries that intersect bbox {bounding_box}. {e}')
        raise


def fetch_admin(bbox=None, admin_level=None, clip=False, h3id_precision=7):
    """
    Parameters
    ----------
    bbox: The bounding box to fetch administrative units for.
    admin_level: The administrative level to fetch. Should be an integer.
    clip: bool, False, if True the admin boundaries are clipped to the bounding box.
    h3id_precision: int, default=7, the tolerance in meters (~11) use to compute the unique id as a h3 hexagon id. Not in use yet.

    Returns
    -------

    """
    with Progress() as progress:
        task = progress.add_task("[cyan]Fetching admin boundaries...", total=5)

        progress.update(task, advance=1, description="[green]Checking intersecting countries")
        intersecting_countries = countries_for_bbox(bounding_box=bbox)
        if not intersecting_countries:
            logger.info(f'The supplied bounding box {bbox} contains no countries')
            progress.update(task, completed=5, description="[red]No countries found in bbox")
            return None

        progress.update(task, advance=1, description="[green]Setting GDAL options")
        options = gdal.VectorTranslateOptions(
            format='FlatGeobuf',
            layerName=f'admin{admin_level}',
            makeValid=True,
            spatFilter=bbox,
        )

        url = f"/vsicurl/https://undpngddlsgeohubdev01.blob.core.windows.net/admin/cgaz/geoBoundariesCGAZ_ADM{admin_level}.fgb"

        configs = {
            'GDAL_DISABLE_READDIR_ON_OPEN': 'TRUE',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.fgb',
            'OGR2OGR_USE_ARROW_API': 'YES'
        }

        progress.update(task, advance=1, description="[green]Downloading and translating with GDAL")
        with gdal.config_options(options=configs):
            with tempfile.TemporaryDirectory(delete=False) as tmpdirname:
                local_path = os.path.join(tmpdirname, f'cgaz_admin{admin_level}.fgb')
                ds = gdal.VectorTranslate(local_path, url, options=options)
                if ds is None:
                    logger.error(f"GDAL VectorTranslate failed for URL: {url} with bbox: {bbox}")
                    progress.update(task, completed=5, description="[red]GDAL VectorTranslate failed")
                    return None
                ds.FlushCache()
                del ds

                progress.update(task, advance=1, description="[green]Reading file with GeoPandas")
                gdf = gpd.read_file(local_path)

                if clip and bbox:
                    progress.update(task, advance=1, description="[green]Clipping to bounding box")
                    bbox_geom = box(*bbox)
                    gdf = gdf.clip(bbox_geom)
                else:
                    progress.update(task, advance=1)
        progress.update(task, completed=5, description="[bold green]Download complete")
        return gdf.to_geo_dict()



if __name__ == "__main__":
    bbox = [22.126465, 2.306506, 32.277832, 8.863362]
    admin_level = 2
    gdf = fetch_admin(bbox=bbox, admin_level=admin_level, clip=True)
    gdf.to_file(f'gpd_admin{admin_level}.fgb', driver='FlatGeobuf')
