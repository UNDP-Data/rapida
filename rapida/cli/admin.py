import asyncio
import logging
import os

from rapida.admin.osm import fetch_admin as fetch_osm_admin, ADMIN_LEVELS
from rapida.admin.ocha import fetch_admin as fetch_ocha_admin
from rapida.admin.cgaz import fetch_admin as fetch_cgaz_admin
from rapida.session import is_rapida_initialized
from rapida.util.bbox_param_type import BboxParamType
from rapida.util.setup_logger import setup_logger
import click
import json
from osgeo import gdal, ogr, osr


gdal.UseExceptions()
logger = logging.getLogger(__name__)

def save(geojson_dict=None, dst_path=None, layer_name=None):
    """
    Save a geojson dict object to OGR
    :param geojson_dict: dict
    :param dst_path: str, path to the dataset name
    :param layer_name: str, the layer name
    :return:
    """
    # if there is a file remove it first to avoid raising a gdal error in cases where the file cannot be overwritten
    if dst_path and os.path.exists(dst_path):
        os.remove(dst_path)

    if dst_path is not None:
        with gdal.OpenEx(json.dumps(geojson_dict, indent=2)) as src:
            src_layer_name = src.GetLayer(0).GetName()
            options = gdal.VectorTranslateOptions(accessMode='overwrite', layerName=layer_name,
                                                 makeValid=True,layers=[src_layer_name], skipFailures=True,
                                                  geometryType='PROMOTE_TO_MULTI')
            ds = gdal.VectorTranslate(destNameOrDestDS=dst_path,
                                      srcDS=src, options=options)
            ds = None


@click.group(short_help=f'fetch administrative boundaries at various levels from OSM/OCHA')
def admin():
    pass


@admin.command(no_args_is_help=True)
@click.argument('destination_path', type=click.Path())
@click.option('-b', '--bbox', required=True, type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north' )
@click.option('-l','--admin_level',
                required=True,
                type=click.IntRange(min=0, max=2, clamp=False),
                help='UNDP admin level from where to extract the admin features'
                )
@click.option('-o', '--osm_level',
              required=False,
              type = int,
                help='OSM admin level from where to extract features corresponding to UNDP admin_level'
              )
@click.option('--clip',

    is_flag=True,
    default=False,
    help="Whether to clip the data to the bounding box."
)
@click.option(
    '--h3id-precision',
    type=int,
    default=7,
    show_default=True,
    help="Precision level for H3 indexing (default is 7)."
)
@click.option('--debug',

    is_flag=True,
    default=False,
    help="Set log level to debug"
)
def osm(bbox=None,admin_level=None, osm_level=None, clip=False, h3id_precision=7, destination_path=None, debug=False,):
    """
    Fetch admin boundaries from OSM

    OSM is a great source of open data and here the Overpass "https://overpass-api.de/api/interpreter" API is used to
    fetch the geospatial representation of administrative divisions. OSM has its own levels described at:

    https://wiki.openstreetmap.org/wiki/Tag:boundary%3Dadministrative#Super-national_administrations

    ranging from 1 (supra national and not rendered) to 12 (noty rendered/reserved).
    In practical terms, the levels start at 2 and end at 10 (inclusive) and can be mapped to  the three UN admin levels
    using following structure:

    {0:2, 1:(3,4), 2:(4,5,6,7,8)}

    The returned admin layer features  the matched admin level and the name of every admin feature that was retrieved by
    the Overpass query. Additionally, all hierarchically superior admin levels and names are returned and attributes of
    the admin units.

    To save the result as a file, for instance, the following command can be executed to extract admin 0 data for Rwanda and Burundi as GeoJSON file:

    rapida admin osm -b "27.767944,-5.063586,31.734009,-0.417477" -l 0 osm.geojson
    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    geojson = fetch_osm_admin(bbox=bbox, admin_level=admin_level,osm_level=osm_level, clip=clip, h3id_precision=h3id_precision)
    if not geojson:
        logger.error('Could not extract admin boundaries from OSM for the provided bbox')
        return
    save(geojson_dict=geojson, dst_path=destination_path, layer_name=f'admin{admin_level}')


@admin.command(no_args_is_help=True)
@click.argument('destination_path', type=click.Path())
@click.option('-b', '--bbox', required=True, type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north' )
@click.option('-l','--admin_level',
                required=True,
                type=click.IntRange(min=0, max=2, clamp=False),
                help='UNDP admin level from where to extract the admin features'
                )

@click.option('--h3id_precision', '-p', default=7, show_default=True, type=int, help='precision of h3id')

@click.option('--clip',

    is_flag=True,
    default=False,
    help="Whether to clip the data to the bounding box."
)

@click.option('--keep-disputed-areas',
    is_flag=True,
    default=False,
    help="Keep admin polygons that are disputed areas with invalid country code"
)
@click.option('--debug',
    is_flag=True,
    default=False,
    help="Set log level to debug"
)
def cgaz(bbox=None,admin_level=None,  clip=False, destination_path=None, debug=False, keep_disputed_areas=None, h3id_precision=None):
    """
    Fetch admin boundaries from CGAZ

    Retrieves administrative boundaries of a specific level/levels that intersect the area covered iby bbox from CGAZ data hosted in Azure Blob Storage
    The admin level argument should be an integer.

    If the -- clip option is supplied, it is going to clip geometries to return only the part of the admin datasets that are within
    the bounding box, if not, it returns whole admin polygons that intersect with the bounding box supplied

    The following is an example to save the result as a geopackage from bbox 33.681335,-0.131836,35.966492,1.158979

    Rapida admin cgaz --bbox 33.681335,-0.131836,35.966492,1.158979 -l 2 --clip /data/admin2_cgaz.gpkg

    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    fetch_cgaz_admin(bbox=bbox, admin_level=admin_level, clip=clip, destination_path=destination_path, keep_disputed_areas=keep_disputed_areas, h3id_precision=h3id_precision)
