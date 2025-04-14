import os
import click
import logging
from rich.progress import Progress
from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger
from cbsurge.project.project import Project


logger = setup_logger()


@click.command(short_help=f'upload a RAPIDA project to Azure file share')
@click.option('-p', '--project',
              default=None,
              type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
              help="Optional. A project folder with rapida.json can be specified. If not, current directory is considered as a project folder.")
@click.option('--max_concurrency',
              default=4,
              show_default=True,
              type=int,
              help=f'The number of threads to use when uploading a file')
@click.option('--overwrite','-o',
              is_flag=True,
              default=False,
              help="Whether to overwrite the project in case it already exists.")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug")
def upload(project=None,max_concurrency=4,overwrite=None, debug: bool =False):
    """
    Upload an entire project folder to Azure File Share

    Usage:

        If you are already in a project folder, run the below command:
        rapida upload

        If you are not in a project folder, run the below command:
        rapida upload --project=<project folder path>

        If a project data already exists in Azure File Share, run the below command:

        rapida upload --overwrite

        To use --overwrite, old project data in Azure File Share will be lost.
    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    if project is None:
        project = os.getcwd()
    else:
        os.chdir(project)
        logger.info(f"Upload command is executed at the project folder: {project}")

    logger.info(f'Going to upload {project} to Azure')
    with Progress() as progress:
        prj = Project(path=project)
        if not prj.is_valid:
            logger.error(f'Project "{project}" is not a valid RAPIDA project')
            return
        prj.upload(progress=progress, overwrite=overwrite, max_concurrency=max_concurrency)
    logger.info(f'Rapida project "{project}" was uploaded successfully to Azure')