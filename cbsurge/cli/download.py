import os
import click
import logging
from cbsurge.az.fileshare import download_project
from rich.progress import Progress

from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger
from cbsurge.project.project import Project

logger = setup_logger()


@click.command(short_help=f'download a RAPIDA project from Azure file share')
@click.argument('project_name', nargs=1,)
@click.argument('destination_folder',
                type=click.Path(exists=False, dir_okay=True, file_okay=False, readable=True, resolve_path=True))
@click.option('--max_concurrency',
              default=4,
              show_default=True,
              type=int,
              help=f'The number of threads to use when downloading a file')
@click.option('--overwrite','-o',
              is_flag=True,
              default=False,
              help="Whether to overwrite the project in case it already exists locally.")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug")
def download(project_name=None, destination_folder=None, max_concurrency=None,overwrite=None, debug: bool =False ):
    """
    Download a project from Azure File Share

    Usage:

        First, please check available projects by using `rapida list` to find a project name to download.

        Then, download a project by using the below command:

        rapida download <project name> <project folder path>

        For example, if project name is `test`, you could run:

        rapida download test ./data

        OR

        rapida download test ./data/test

        The project data will be downloaded under ./data/test folder.

        If a project data already exists in local, run the below command:

        rapida download <project name> <project folder path> --overwrite

        To use --overwrite, old project data in local storage will be lost..
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
            if not overwrite:
                logger.warning(f'Project "{project_name}" already exists in {project_path}. To overwrite it use the --overwrite flag')
                return
            else:
                logger.warning(f'Project "{project_name}" already exists in {project_path}. it will be overwritten by downloaded files')
                pass
        else:
            if not overwrite:
                logger.warning(f'Folder "{project_name}" exists, but is not a valid RAPIDA project. To force downloading it use the --overwrite flag')
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
                         overwrite=overwrite,
                         max_concurrency=max_concurrency)
    logger.info(f'Project "{project_name}" was downloaded successfully to {project_path}')