import logging
import numbers
from datetime import date
import click
from rapida.cli import RapidaCommandGroup
from rapida.ntl.science.const import ARCHIVE, OPERATIONAL, PROCESSING_LEVEL_NAMES
from rapida.ntl.science.search import search as _search
from rapida.util.bbox_param_type import BboxParamType
from rich.table import Table
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


@click.group(cls=RapidaCommandGroup)



def ntl():
    """Nighttime Lights VIIRS data and impact detection"""
    pass
@ntl.group(short_help=f'Search for available NTL data products across tiers and streams')
def search():
    """Search for available NTL data products across distinct data streams."""
    pass

@search.command(name='noaa', short_help=f'Search for available NTL data from operational NOAA stream')
async def search_noaa( ):
    logger.info('searching for operational NOAA NTL data')


@search.command(name='nasa', short_help=f'Search for available NTL data from NASA science archive stream')


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

@click.pass_context
def search_nasa(ctx, bbox:tuple[numbers.Number]=None, target_date:date=None, stream:str = None, processing_level:str=None):

    progress = ctx.obj.get('progress')

    urls = _search(processing_level=processing_level,target_date=target_date,
                  bbox=bbox,stream=stream, progress=progress)

    if urls:
        table = Table(title=f" {processing_level} VIIRS satellites tiles for the night of  {target_date.date()} covering {bbox}",
                      title_style="bold yellow")
        table.add_column("Product", style="red", justify='center')
        table.add_column("URI", style="green", justify='center')
        for url in urls:
            table.add_row(*url)
        progress.console.print(table)
@ntl.command(short_help=f'Download selected NTL data')
async def download():
    logger.info('Downloading NTL')

@ntl.command(short_help=f'Execute crisis impact detection (48h Alerts / 72h Assessments)')
async def detect():
    logger.info('Detecting impact on the ground')


@ntl.command(short_help=f'Track long-term resilience and recovery curves (2-3 Week horizon)')
async def monitor():
    logger.info('Monitoring recovery')


