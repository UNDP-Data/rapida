from typing import Union

import click
import logging
import tempfile
from rapida.util.bbox_param_type import BboxParamType
from rapida.connectivity import run_connectivity_analysis
from rapida.cli.aclick import AsyncCommand
from rapida.connectivity.isochrone import MODE_MAP

logger = logging.getLogger(__name__)


def parse_intervals(ctx, param, value):
    """Parses a comma-separated string of numbers into a list of integers."""
    if not value:
        return [5, 15, 30, 60]  # Fallback default

    try:
        # Split by comma, strip spaces, and cast to int
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise click.BadParameter("Time intervals must be a comma-separated list of integers (e.g., 5,15,30,60).")

@click.command(short_help='run connectivity analysis', cls=AsyncCommand)

@click.option('-b', '--bbox',
              required=True,
              type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north'
              )
@click.option(
        '-m', "--mode", "travel_mode",
        type=click.Choice(MODE_MAP, case_sensitive=False),
        default='walk',
        required=True,
        help=f"The means of travel when delineating isochrones"

    )

@click.option(
    '-ti', '--time-intervals',
    type=str,
    default="5,15,30,60",
    callback=parse_intervals,
    help="Comma-separated time intervals in minutes for the catchment areas."
)

@click.option(
    '-bd', '--barriers-dataset',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=None,
    help="Path to an OGR-supported vector data source (e.g., GPKG, Shapefile) containing exclusion zones."
)
@click.option(
    '-bl', '--barriers-layer',
    type=str,
    default="0",
    help="Name or index of the layer to use from barriers dataset. Defaults to the first layer (layer 0)."
)

@click.option('-bb', "--barriers-buffer",
    type=int,
    default=5,
    required=False,
    help="The value in meters to used to buffer the geometries in barriers/dataset/layer in case the barriers are lines"
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
    help="Destination directory to save the downloaded OSM pbf files."
)


@click.pass_context
async def connectivity(ctx, bbox:tuple[float, float, float, float]=None, travel_mode:str=None,
                       time_intervals:list[int] =None, dst_dir:str=None,
                       barriers_dataset:str=None, barriers_layer:str=None, barriers_buffer:int=None
    ):
    logger.info(f'Running connectivity analysis ')
    progress = ctx.obj.get('progress')
    return await run_connectivity_analysis(
        bbox=bbox, dst_dir=dst_dir, travel_mode=travel_mode, time_intervals=time_intervals,
        barriers_dataset=barriers_dataset, barriers_layer=barriers_layer, barriers_buffer=barriers_buffer, progress=progress
    )