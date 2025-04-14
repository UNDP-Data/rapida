import logging
import os
import click
import sys
from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger
from cbsurge.project.project import Project

logger = logging.getLogger(__name__)


@click.command(no_args_is_help=True, short_help='create a RAPIDA project in a new folder')
@click.option('-n', '--name', required=True, type=str,
              help='Name representing a new folder in the current directory' )
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
def create(name=None, polygons=None, mask=None, comment=None, debug=False):
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    abs_folder = os.path.abspath(name)
    if os.path.exists(abs_folder):
        logger.error(f'Folder "{name}" already exists in {os.getcwd()}')
        sys.exit(1)
    else:
        os.mkdir(abs_folder)

    project = Project(path=abs_folder, polygons=polygons, mask=mask, comment=comment)
    assert project.is_valid
    logger.info(f'Project "{project.name}" was created successfully.')