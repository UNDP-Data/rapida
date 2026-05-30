import logging
import numbers
import os.path
from datetime import datetime
from typing import Iterable
import click
import tempfile
from rapida.cli import RapidaCommandGroup
from rapida.ntl.nasa.const import ARCHIVE, OPERATIONAL, PROCESSING_LEVEL_NAMES, PRODUCT_NAMES, PRODUCTS, \
    NTL_FILENAME_PATTERN, ROUTES, COLLECTIONS
from rapida.ntl.nasa.search import search as nasa_search
from rapida.ntl.noaa.search import async_search_granules, VIIRSNavigator
from rapida.util.bbox_param_type import BboxParamType
from rapida.ntl.nasa.io import download as download_from_nasa, bulk_download
from rapida.ntl.noaa.const import SOURCE_NAMES, PRODUCT_NAMES as OPER_PRODUCT_NAMES
from rapida.ntl.noaa.io import download as download_from_noaa, bytesto
from rich.table import Table
from rapida.ntl.nasa.io import bulk_download as bdownload
from rapida.project.project import Project

logger = logging.getLogger(__name__)


class ProcessingLevelChoiceOption(click.Option):
    """
    Custom Click option that dynamically validates choices based on the value
    of a companion option (in this case, '--stream').
    """

    def handle_parse_result(self, ctx, opts, args):
        # Retrieve the value of 'stream' that click has already processed
        stream_val = opts.get('stream')

        # If stream is found, dynamically inject its specific valid levels into the choice validator
        if stream_val:
            stream_key = stream_val.upper()  # Handle casing normalization
            valid_choices = PROCESSING_LEVEL_NAMES.get(stream_key, [])
            # Map choice array to lowercase to keep user typing natural
            self.type = click.Choice([c.upper() for c in valid_choices], case_sensitive=False)

        return super().handle_parse_result(ctx, opts, args)


def validate_products_strict(ctx, param, value):
    if not value:
        return value

    # Check for mixed catalogs
    has_nrt = any('nrt' in p.lower() for p in value)
    has_std = any('nrt' not in p.lower() for p in value)
    nrt_choices  = [item for p in COLLECTIONS['LANCEMODIS'].values() for item in p]
    std_choices = [item for p in COLLECTIONS['LAADS'].values() for item in p]
    if has_nrt and has_std:
        raise click.BadParameter(
            f"Cannot mix NRT and Standard products in the same command. "
            f"They belong to different catalogs:\n"
            f"LANCEMODIS - NOAA operational: {', '.join(nrt_choices)}\n" 
            f"LAADS - NASA archive : {', '.join(std_choices)}"
        )

    if has_nrt:
        return tuple(nrt_choices)
    else:
        return tuple(std_choices)

class NASAProductsChoiceOption(click.Option):
    """
    Custom Click option that dynamically validates choices
    """

    def handle_parse_result(self, ctx, opts, args):
        # Retrieve the value of 'stream' that click has already processed
        products = opts.get('products')
        has_nrt = 'nrt' in products[0].lower()
        if has_nrt:
            valid_choices = [item for p in COLLECTIONS['LANCEMODIS'].values() for item in p]

        else:
            valid_choices = [item for p  in COLLECTIONS['LAADS'].values() for item in p]

        self.type = click.Choice([c.upper() for c in valid_choices], case_sensitive=False)
        return super().handle_parse_result(ctx, opts, args)


@click.group(cls=RapidaCommandGroup)


def ntl():
    """Nighttime Lights VIIRS data and impact detection"""
    pass

@ntl.group(short_help=f'Search for available NTL data')
def search():
    """Search for available NTL data products across distinct data streams."""
    pass

@search.command(name='noaa', short_help=f'Search for available  NTL operational data from NOAA source')

@click.option('-b', '--bbox',
              required=True,
              type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north'
              )
@click.option("--date", "target_date",
              type=click.DateTime(formats=["%Y-%m-%d"]),
              required=True,
              help=''
              )

@click.option(
    "--sat",
    "-s",
    "satellites", # This will be the name of the argument in your function
    type=click.Choice(VIIRSNavigator.SATELLITES, case_sensitive=False),
    multiple=True,
    default=list(VIIRSNavigator.SATELLITES),
    help=f"Target satellite(s). Use multiple times for more than one ({','.join(VIIRSNavigator.SATELLITES)})."
)


@click.option(
    '--cmask', '-cm', "cmask",
    is_flag=True,
    help=(
        "Enable Cloud Mask optimization by favouring  "
        "the granules where the target bbox is mostly cloud free. "
        "If omitted, defaults to standard Geometry filtering (elevation >20°, offset <1500km)."
    )
)




@click.pass_context
async def search_noaa(ctx, bbox:tuple[numbers.Number]=None, target_date:datetime=None, satellites:list[str] = [], cmask:bool=None  ):

    progress = ctx.obj.get('progress')
    table = Table(title=f"VIIRS satellites granules for the night of  {target_date.date()} covering {bbox}",
                  title_style="bold yellow")
    table.add_column("Position", justify="center", style="white")
    table.add_column("Satellite", style="green", justify='center')
    table.add_column("Timestamp (UTC)", style="cyan", justify='center')
    # table.add_column("Scan Start Date and Time (UTC)", style="red", justify='center')
    table.add_column("Bbox offset from SSP (km)", justify="center", style="white")
    table.add_column("Elevation above bbox (degrees)", justify="center", style="white")
    if cmask:
        table.add_column("Cloud coverage in bbox (%)", justify="center", style="white")
    table.add_column("Score (%)", justify="center", style="white")
    table.add_column("BBOX intersection (%)", justify="center", style="white")

    granules = await async_search_granules(
        satellites=satellites, target_date=target_date, bbox=bbox,
        cmask=cmask, progress=progress)
    if granules:
        for i, granule in enumerate(granules, start=1):
            if cmask:
                values = f'{i}', granule.sat, granule.timestamp, f'{granule.offset}', f'{granule.elevation:.2f}', f'{granule.cloud_cover}', f'{granule.rank}', f'{granule.pint}'
            else:
                values = f'{i}', granule.sat, granule.timestamp, f'{granule.offset}', f'{granule.elevation:.2f}', f'{granule.rank}', f'{granule.pint}'
            table.add_row(*values)

    if table.row_count == 0:
        progress.console.print("[bold red]No granules found for this criteria.[/bold red]")
    else:
        progress.console.print(table)
        progress.console.print(f"\n[dim]Note: Each granule represents {1025 / 12:.2f}s of instrument data.[/dim]")


@search.command(name='nasa', short_help=f'Search for available NTL science data from NASA source')

@click.option('-b', '--bbox',
              required=True,
              type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north'
              )
@click.option("--date", "nominal_date",
              type=click.DateTime(formats=["%Y-%m-%d"]),
              required=True,
              help='The human experience of a specific night, local time zone matched to the center of bbox'
              )
@click.option(
        '-s', '--stream',
        type=click.Choice((ARCHIVE, OPERATIONAL), case_sensitive=False),
        required=True,
        help=f"NASA data stream tier: '{OPERATIONAL}' for immediate low-latency processing, '{ARCHIVE}' for refined archives."
    )

@click.option(
        '-l', '--processing-level',
        cls=ProcessingLevelChoiceOption,
        required=True,
        help=(
                "NASA data stream processing level. Available options depend on selection of --stream. "
                "For NRT: [a1, a1g, a2]. For ARCHIVE: [a1, a2, a3, a4]."
            )
    )

@click.option(
        '-r', '--route',
        type=click.Choice(ROUTES, case_sensitive=False),
        default='API',
        required=True,
        help=f"Route to use when searching for data. Options are STAC for CMR STAC or API for NASA blackmarble API"

    )

@click.pass_context
def search_nasa(ctx, bbox:tuple[numbers.Number]=None, nominal_date:datetime=None, stream:str = None, processing_level:str=None, route:str=None):

    progress = ctx.obj.get('progress')

    urls = nasa_search(processing_level=processing_level, nominal_date=nominal_date,
                       bbox=bbox, stream=stream, route=route, progress=progress, push_to_cache=True)

    if urls:
        table = Table(title=f" {processing_level} VIIRS satellites tiles for the night of  {nominal_date.date()}-{nominal_date.strftime('%Y%j')} covering {bbox}",
                      title_style="bold yellow")
        table.add_column("Product", style="red", justify='center')
        table.add_column("Timestamp", style="red", justify='center')
        table.add_column("Tile", style="red", justify='center')
        table.add_column("URI", style="green", justify='center')
        for e in urls:
            table.add_row(*e)
        progress.console.print(table)


@ntl.group(short_help=f'Download NTL data ')
def download():
    pass



@download.command(name='nasa', short_help=f'Download NTL products from NASA')


@click.option( "-t", "--timestamp", "timestamp",
               type=str,
               required=True,
               help='Granule timestamp string as date and time. Ex: 202604152232 '
               )
@click.option(
    "-p",
                "product",
                type=click.Choice(PRODUCT_NAMES, case_sensitive=True),
                required=True,
                help=f'The product to download.'

    )
@click.option('--tile',
              required=False,
              type=str,
              help='A specific tile number conforming to NASA BalckMarble 10x10 degrres tile numbering. Ex: h21v03 '
              )

@click.option(
    "--dst-dir",
    "dst_dir",     # Function argument name
    type=click.Path(
        exists=False,      # Set to True if you want Click to fail if the dir doesn't exist yet
        file_okay=False,   # Strictly enforce that this is a directory, not a file
        dir_okay=True,
        resolve_path=True  # Resolves relative paths (like '.') to absolute paths automatically
    ),
    default=tempfile.gettempdir(),           # Defaults to the current working directory
    show_default=True,     # Tells the user what the default is in the --help menu
    help="Destination directory to save the downloaded the images."
)


@click.pass_context
async def download_nasa(ctx, timestamp:str = None, product:str=None, tile:str=None, dst_dir:str=None):
    progress = ctx.obj.get('progress')

    downloaded_files = await download_from_nasa(timestamp=timestamp, product=product, tile=tile, dst_dir=dst_dir,progress=progress)

    if downloaded_files:
        table = Table(title=f"Downloaded files for {product.upper()} {timestamp} ", title_style="bold yellow")

        table.add_column("Path", style="red", justify='center')
        table.add_column("Timestamp", style="red", justify='center')
        table.add_column("Tile", style="red", justify='center')
        table.add_column("Size", style="green", justify='center')
        for local_file_path in downloaded_files:
            _, file_name = os.path.split(local_file_path)
            file_size = os.path.getsize(local_file_path)
            m = NTL_FILENAME_PATTERN.match(file_name)
            meta = m.groupdict()
            tile = meta['tile']
            table.add_row(local_file_path, timestamp, tile, f'{file_size}')

        progress.console.print(table)


@download.command(name='noaa', short_help=f'Download operational NTL data from NOAA')
@click.option(
    "--sat", "-s",
    "satellite", # This will be the name of the argument in your function
    type=click.Choice(VIIRSNavigator.SATELLITES, case_sensitive=False),
    multiple=False,
    help=f"Target satellite(s). One of ({','.join(VIIRSNavigator.SATELLITES)}) that produced the granule."
)

@click.option("--timestamp", "-t", "timestamp", type=str, required=True, help='Granule timestamp string as date and time. Ex: 202604152232 ')
@click.option(
    "--products",
                "-p",
                "products",
                type=click.Choice(OPER_PRODUCT_NAMES, case_sensitive=False),
                default=OPER_PRODUCT_NAMES,
                multiple=True,
                required=False,
                help=f'One or more of the operational products: {",".join(OPER_PRODUCT_NAMES)} to download.'
    )
@click.option("-src", '--src', "source",
              type=click.Choice(SOURCE_NAMES, case_sensitive=False),
              required=False,
              help='The source {AMAZON/GOOGLE} where to search for the granules.'
              )
@click.option(
    "--dst-dir",
    "-d",           # Short option
    "dst_dir",     # Function argument name
    type=click.Path(
        exists=False,      # Set to True if you want Click to fail if the dir doesn't exist yet
        file_okay=False,   # Strictly enforce that this is a directory, not a file
        dir_okay=True,
        resolve_path=True  # Resolves relative paths (like '.') to absolute paths automatically
    ),
    default=tempfile.gettempdir(),   # Defaults to the current working directory
    show_default=True,     # Tells the user what the default is in the --help menu
    help="Destination directory to save the downloaded the images."
)

@click.pass_context
async def download_noaa(ctx, satellite:str=None, timestamp:str=None, products:Iterable[str]=None, source:str=None, dst_dir:str=None ):
    progress = ctx.obj.get('progress')
    downloaded_files = await download_from_noaa(satellite=satellite,timestamp=timestamp,
                           source=source, products=products,dest_dir=dst_dir, progress=progress)

    if downloaded_files:
        table = Table(title=f"VIIRS satellites images for the night of  {timestamp} ",
                      title_style="bold yellow")

        table.add_column("Satellite", style="green", justify='center')
        table.add_column("Timestamp (UTC)", style="cyan", justify='center')
        table.add_column("Downloaded file", justify="left", style="red")
        table.add_column("File size", justify="center", style="white")

        for _, local_file_path, file_size in downloaded_files:
            values = satellite, timestamp, f'{local_file_path}', f'{bytesto(file_size, "m"):.2f} MB'
            table.add_row(*values)
        progress.console.print(table)




@download.command(name='bulk', short_help=f'Download daily  NTL products from NASA BlackMarble using STAC ')

@click.option('-b', '--bbox',
              required=True,
              type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north'
              )
@click.option("--from", "start_date",
              type=click.DateTime(formats=["%Y-%m-%d"]),
              required=True,
              help='The start date of required period'
              )
@click.option("--to", "end_date",
              type=click.DateTime(formats=["%Y-%m-%d"]),
              required=True,
              help='The end date of required period'
              )

@click.option(
        '-p','--products',
        # type=click.Choice(PRODUCTS, case_sensitive=False),
        cls=NASAProductsChoiceOption,
        #callback=validate_products_strict,
        required=True,
        multiple=True,
        help=f"One or more STAC collections hosting different processing level or products limited to one stream. "
    )
@click.option(
    "--dst-dir",
    "dst_dir",     # Function argument name
    type=click.Path(
        exists=False,      # Set to True if you want Click to fail if the dir doesn't exist yet
        file_okay=False,   # Strictly enforce that this is a directory, not a file
        dir_okay=True,
        resolve_path=True  # Resolves relative paths (like '.') to absolute paths automatically
    ),
    default=tempfile.gettempdir(),           # Defaults to the current working directory
    show_default=True,     # Tells the user what the default is in the --help menu
    help="Destination directory to save the downloaded the images."
)


@click.pass_context
async def bulk_download(ctx, bbox:tuple[numbers.Number]=None, start_date:datetime=None, end_date:datetime=None,
                            products:str=None, dst_dir:str=None):
    progress = ctx.obj.get('progress')

    if start_date > end_date:
        raise click.UsageError(f'--from {start_date} must be smaller or equal then --to {end_date}')

    if 'nrt' in products[0].lower():
        stream = OPERATIONAL
        now = datetime.now()

        start_days_difference = abs((now - start_date).days)
        if start_days_difference > 7:
            raise ValueError(f'Invalid start_date={start_date}.{stream} stream holds max 7 days of data. ')
        end_days_difference = abs((now - end_date).days)
        if end_days_difference > 7:
            raise ValueError(f'Invalid start_date={end_date}.{stream} stream holds max 7 days of data. ')

    else:
        stream = ARCHIVE
    downloaded_files = await bdownload(
        bbox=bbox, start_date=start_date, end_date=end_date,
        stream=stream, products=products,
        dst_dir=dst_dir,progress=progress
    )


@ntl.command(short_help=f'Find and download best NTL data for a specific area and time ')

@click.option('-b', '--bbox',
              required=True,
              type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north'
              )
@click.option("--from", "start_date",
              type=click.DateTime(formats=["%Y-%m-%d"]),
              required=True,
              help='The start date of required period'
              )
@click.option("--to", "end_date",
              type=click.DateTime(formats=["%Y-%m-%d"]),
              required=True,
              help='The end date of required period'
              )

@click.option(
        '-p','--products',
        # type=click.Choice(PRODUCTS, case_sensitive=False),
        cls=NASAProductsChoiceOption,
        #callback=validate_products_strict,
        required=True,
        multiple=True,
        help=f"One or more STAC collections hosting different processing level or products limited to one stream. "
    )
@click.option(
    "--dst-dir",
    "dst_dir",     # Function argument name
    type=click.Path(
        exists=False,      # Set to True if you want Click to fail if the dir doesn't exist yet
        file_okay=False,   # Strictly enforce that this is a directory, not a file
        dir_okay=True,
        resolve_path=True  # Resolves relative paths (like '.') to absolute paths automatically
    ),
    default=tempfile.gettempdir(),           # Defaults to the current working directory
    show_default=True,     # Tells the user what the default is in the --help menu
    help="Destination directory to save the downloaded the images."
)


@click.pass_context
async def fetch(ctx):
    progress = ctx.obj.get('progress')



@ntl.command(short_help=f'Execute crisis impact detection (48h Alerts / 72h Assessments)')
@click.pass_context
async def detect(ctx):
    logger.info('Detecting impact on the ground')

#
# @ntl.command(short_help=f'Track long-term resilience and recovery curves (2-3 Week horizon)')
# async def monitor():
#     logger.info('Monitoring recovery')


