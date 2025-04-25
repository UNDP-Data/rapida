import os
import click
import logging
from rapida.az.fileshare import download_project
from rich.progress import Progress

from rapida.session import is_rapida_initialized
from rapida.util.setup_logger import setup_logger
from rapida.project.project import Project

logger = logging.getLogger(__name__)


@click.command(short_help=f'download a RAPIDA project from Azure file share', no_args_is_help=True)
@click.argument('project_name', nargs=1,)
@click.argument('destination_folder',
                type=click.Path(exists=False, dir_okay=True, file_okay=False, readable=True, resolve_path=True))
@click.option('--max_concurrency', '-c',
              default=4,
              show_default=True,
              type=int,
              help=f'The number of threads to use when downloading a file')
@click.option('--force','-f',
              is_flag=True,
              default=False,
              help="Whether to overwrite the project in case it already exists locally.")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug")
def download(project_name=None, destination_folder=None, max_concurrency=None,force=None, debug: bool =False ):
    """
    Download a project from Azure File Share.

    Firstly, please check available projects by using `rapida list` to find a project name to download.

    Usage:

    rapida download <project name> <project folder path>

    For example, the project data will be downloaded under ./data/test folder if a project name is `test` if the below command is executed.

    Example:

        rapida download test ./data

        rapida download test ./data/test



    To use `-f/--force`, project data will be overwritten if it already exists in local storage.

    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    last_folder = destination_folder.split('/')[-1]
    project_path = destination_folder
    if project_name != last_folder:
        project_path = os.path.join(destination_folder, project_name)

    if os.path.exists(project_path):
        project = Project(path=project_path)
        if project.is_valid:
            if not force:
                logger.warning(f'Project "{project_name}" already exists in {project_path}. To overwrite it use the --force flag')
                return
            else:
                logger.warning(f'Project "{project_name}" already exists in {project_path}. it will be overwritten by downloaded files')
                pass
        else:
            if not force:
                logger.warning(f'Folder "{project_name}" exists, but is not a valid RAPIDA project. To force downloading it use the --force flag')
            else:
                logger.warning(
                    f'Folder "{project_name}" exists, but is not a valid RAPIDA project. it will be overwritten by downloaded files')
                pass

    if not os.path.exists(project_path):
        os.makedirs(project_path)

    logger.info(f'Going to download rapida project "{project_name}" from Azure')
    with Progress() as progress:
        download_project(name=project_name,
                         dst_folder=project_path,
                         progress=progress,
                         overwrite=force,
                         max_concurrency=max_concurrency)
    logger.info(f'Project "{project_name}" was downloaded successfully to {project_path}')