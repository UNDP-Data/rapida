import click
import logging
import tempfile
from rapida.util.bbox_param_type import BboxParamType
from rapida.connectivity import run_connectivity_analysis
from rapida.cli.aclick import AsyncCommand


logger = logging.getLogger(__name__)


@click.command(short_help='run connectivity analysis', cls=AsyncCommand)

@click.option('-b', '--bbox',
              required=True,
              type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north'
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
async def connectivity(ctx, bbox:tuple[float, float, float, float]=None, dst_dir:str=None ):
    logger.info(f'Running connectivity analysis ')
    progress = ctx.obj.get('progress')
    return await run_connectivity_analysis(bbox=bbox, dst_dir=dst_dir, progress=progress)