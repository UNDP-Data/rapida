from email.policy import default

import click
from cbsurge.exposure.builtenv.buildings.fgb import download_bbox, download_admin
from cbsurge.exposure.builtenv.buildings.pmt import download as download_pmt, GMOSM_BUILDINGS
from cbsurge.util import BboxParamType
import asyncio

@click.group()
def buildings():
    f"""Command line interface for {__package__} package"""
    pass

@click.group()
def download():
    f"""Command line interface for {__package__} package"""
    pass


@download.command(no_args_is_help=True)

@click.option('-b', '--bbox', required=True, type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north' )
@click.option('-o', '--out-path', required=True, type=click.Path(),
              help='Full path to the buildings dataset' )

@click.option('-bs', '--batch-size', type=int, default=65535,
              help='The max number of buildings to be downloaded in one chunk or batch. '
                   )
def fgbbbox(bbox=None, out_path=None, batch_size:[int,None]=1000):
    """
        Download/stream buildings from VIDA buildings using pyogrio/pyarrow API

        :param bbox: iterable of floats, xmin, ymin, xmax,ymax
        :param out_path: str, full path where the buildings layer will be written
        :param batch_size: int, default=1000, the max number of buildings to download in one batch
        If supplied, the buildings are downloaded in batches otherwise they are streamd through pyarrow library

    """
    download_bbox(bbox=bbox, out_path=out_path, batch_size=batch_size)



@download.command(no_args_is_help=True)
@click.option('-a', '--admin-path', required=True, type=click.Path(),
              help='Full path to the admin dataset' )
@click.option('-o', '--out-path', required=True, type=click.Path(),
              help='Full path to the buildings dataset' )
@click.option('--country-col-name', required=True, type=str,
              help='The name of the column from the admin layer attributes that contains the ISO3 country code' )
@click.option('--admin-col-name', required=True, type=str,
              help='The name of the column from the admin layer attributes that contains the admin unit name' )
@click.option('-bs', '--batch-size', type=int, default=65535,
              help='The max number of buildings to be dowloaded in one chunk or batch. ')

def fgbadmin(admin_path=None, out_path=None, country_col_name=None, admin_col_name=None, batch_size=None):

    """Fetch buildings from VIDA FGB based on the boundaries of the admin units"""


    download_admin(admin_path=admin_path,out_path=out_path,
                   country_col_name=country_col_name, admin_col_name=admin_col_name, batch_size=batch_size)



@download.command(no_args_is_help=True)

@click.option('-b', '--bbox', required=True, type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north' )
@click.option('-o', '--out-path', required=True, type=click.Path(),
              help='Full path to the buildings dataset' )

@click.option('-z', '--zoom-level', type=int,
              help='The zoom level from which to fetch the buildings.Defaults to dataset max zoom-1')
@click.option('-x', type=int,
              help='The x coordinate of the til;e for a specific zoom level')
@click.option('-y', type=int,
              help='The y coordinate of the til;e for a specific zoom level')

def pmt(bbox=None, out_path=None, zoom_level=None,  x=None,y=None):
    """
        
        Fetch buildings from url remote source in PMTiles format from VIDA
        https://data.source.coop/vida/google-microsoft-osm-open-buildings/pmtiles/goog_msft_osm.pmtiles

        :param bbox: iterable of floats, xmin, ymin, xmax,ymax
        :param out_path: str, full path where the buildings layer will be written
        :param zoom_level: int the zoom level, defaults to max zoom level -1
        :param x:int,  the tile x coordinate at a specific zoom level, can be used as a filter
        :param y: int, the tile y coordinate at a specific zoom level, can be sued as a filter
        
    """
    asyncio.run(download_pmt(bbox=bbox,out_path=out_path, zoom_level=zoom_level, x=x, y=y))

buildings.add_command(download)