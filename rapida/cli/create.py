import logging
import os
import click
import sys
from rapida.session import is_rapida_initialized
from rapida.util.setup_logger import setup_logger
from rapida.project.project import Project

logger = logging.getLogger(__name__)


@click.command(no_args_is_help=True, short_help='create a RAPIDA project in a new folder')
@click.argument('project_name', type=str, required=True)
@click.argument('destination_folder', type=click.Path(), required=False)
@click.option('-p', '--polygons', required=True, type=str,
              help='Full path to the vector polygons dataset in any OGR supported format' )
@click.option('-m', '--mask', required=False, type=str,
              help='Full path to the mask dataset in any GDAL/OGR supported format. Can be vector or raster' )
@click.option('-c', '--comment', required=False, type=str,
              help='Any comment you might want to add into the project config' )
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug"
              )
def create(project_name: str, destination_folder: str = None, polygons=None, mask=None, comment=None, debug=False):
    """
    Create a RAPIDA project in a new folder. As default, it creates a project folder with the same name as user specified.

    Usage:

    rapida create [project name] --polygons /path/to/vector_polygons.gpkg --mask /path/to/mask.gpkg

        This creates a project folder at current working directory with project polygon of a given GeoPackage.

        If --mask is provided, it creates a mask layer in the project database, and assessment can compute affected version of variables.

    rapida create [project name] [destination folder] --polygons /path/to/vector_polygons.gpkg

        If you set a folder path in the second positional argument, it creates a project folder at a given destination folder.

        Second positional argument is optional. If skipped, current directory is used.

    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    project_folder = os.path.abspath(project_name)
    if destination_folder is not None:
        parent_folder = os.path.abspath(destination_folder)
        project_folder = os.path.join(parent_folder, project_name)

    if os.path.exists(project_folder):
        logger.error(f'Folder "{project_name}" already exists in {os.getcwd()}')
        sys.exit(1)
    else:
        os.makedirs(project_folder)

    project = Project(path=project_folder, polygons=polygons, mask=mask, comment=comment)
    assert project.is_valid
    logger.info(f'Project "{project.name}" was created successfully.')