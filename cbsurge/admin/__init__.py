# from cbsurge.admin.ocha import ARCGIS_SERVER_ROOT
# from cbsurge.admin.ocha import OCHA_COD_ARCGIS_SERVER_ROOT
# from cbsurge.admin.ocha import http_get_json
# from cbsurge.admin.osm import OVERPASS_API_URL
# import asyncio
#
# asyncio.run(http_get_json(ARCGIS_SERVER_ROOT, timeout=10))
# asyncio.run(http_get_json(OCHA_COD_ARCGIS_SERVER_ROOT, timeout=10))
# asyncio.run(http_get_json(OVERPASS_API_URL, timeout=10))
#
from random import choices

from cbsurge.admin.osm import fetch_admin as fetch_osm_admin, ADMIN_LEVELS
from cbsurge.admin.ocha import fetch_admin as fetch_ocha_admin
import click
import json


class BboxParamType(click.ParamType):
    name = "bbox"

    def convert(self, value, param, ctx):
        try:
            bbox = [float(x.strip()) for x in value.split(",")]
            fail = False
        except ValueError:  # ValueError raised when passing non-numbers to float()
            fail = True

        if fail or len(bbox) != 4:
            self.fail(
                f"bbox must be 4 floating point numbers separated by commas. Got '{value}'"
            )

        return bbox


@click.group()
def admin():
    f"""Command line interface for {__package__} package"""
    pass


@admin.command(no_args_is_help=True)
@click.option('-b', '--bbox', required=True, type=BboxParamType(), help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north')
@click.option('-l','--admin_level',
                required=True,
                type=click.Choice(choices=['0','1','2'], case_sensitive=False),
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

def osm(bbox=None,admin_level=None, osm_level=None, clip=False, h3id_precision=7):
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

    python -m  cbsurge.cli admin osm -b "27.767944,-5.063586,31.734009,-0.417477" -l 0 > osm.geojson
    """
    geojson = fetch_osm_admin(bbox=bbox, admin_level=admin_level,osm_level=osm_level, clip=clip, h3id_precision=h3id_precision)
    if geojson:
        click.echo(json.dumps(geojson))

@admin.command(no_args_is_help=True)
@click.option('-b', '--bbox', required=True, type=BboxParamType(), help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north')
@click.option('-l','--admin_level',
                required=True,
                type=click.Choice(choices=['0','1','2'], case_sensitive=False),
                help='UNDP admin level from where to extract the admin features'
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

def ocha(bbox=None,admin_level=None,  clip=False, h3id_precision=7):
    """
    Fetch admin boundaries from OCHA COD

    Retrieves administrative boundaries of a specific level/levels that intersect the area covered iby bbox from
    OCHA COD database hosted on ArcGIS server
    The admin_level argument can be an integer or a dictionary.

    In case an integer is supplied, the best effort is made to retrieve admin boundaries from that specific admin
    level data layer for all countries  whose extent intersects the bounding box

    The OCA COD database is hosted on an ArcGIS server instance and is organized on a per country basis. As a result
    this function also uses a per country approach to fetch features and merge them into one layer in case the supplied
    bounding box covers several countries.

    To save the result as a file, for instance, the following command can be executed to extract admin 0 data for Rwanda and Burundi as GeoJSON file:

    python -m  cbsurge.cli admin ocha -b "27.767944,-5.063586,31.734009,-0.417477" -l 0 > ocha.geojson
    """
    geojson = fetch_ocha_admin(bbox=bbox, admin_level=admin_level, clip=clip, h3id_precision=h3id_precision)
    if geojson:
        click.echo(json.dumps(geojson))
