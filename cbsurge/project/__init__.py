
import os
import click
import logging
import sys
from cbsurge.az.fileshare import list_projects, download_project
from rich.progress import Progress

from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger
from cbsurge.project.project import Project

logger = setup_logger()


@click.command(no_args_is_help=True, short_help='create a RAPIDA project in a new folder')
@click.option('-n', '--name', required=True, type=str,
              help='Name representing a new folder in the current directory' )
@click.option('-p', '--polygons', required=True, type=str,
              help='Full path to the vector polygons dataset in any OGR supported format' )
@click.option('-m', '--mask', required=False, type=str,
              help='Full path to the mask dataset in any GDAL/OGR supported format. Can be vector or raster' )
@click.option('-c', '--comment', required=False, type=str,
              help='Any comment you might want to add into the project config' )

def create(name=None, polygons=None, mask=None, comment=None):

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


@click.command(short_help=f'list RAPIDA projects/folders located in default Azure file share')
def list():
    if not is_rapida_initialized():
        return

    const = '-'*15
    tabs = '\t'*1
    click.echo(f'{const} Available RAPIDA projects {const}')
    for project_name in list_projects():
        click.echo(f'{tabs}"{project_name}"')




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


@click.command(short_help=f'delete a RAPIDA project from Azure file share')
@click.option('-p', '--project',
              default=None,
              type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
              help="Optional. A project folder with rapida.json can be specified. If not, current directory is considered as a project folder.")
@click.option('-y', '--yes',
              is_flag=True,
              default=False,
              help="Optional. If True, it will automatically answer yes to prompts. Default is False.")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug")
def delete(project: str, yes: bool = False, debug: bool =False):
    """
    Delete a project from local storage and Azure File Share if it was uploaded.

    Usage:

        If you are already in a project folder, run the below command:
        rapida delete

        If you are not in a project folder, run the below command:
        rapida delete --project=<project folder path>

        This command shows prompts to confirm deletion of a project to prevent deleting accidentally. If you wish to answer all prompts to yes, use -y or --yes in the command:

        rapida delete --yes
    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    if project is None:
        project = os.getcwd()
    else:
        os.chdir(project)
        logger.info(f"Delete command is executed at the project folder: {project}")
    prj = Project(path=project)
    if not prj.is_valid:
        logger.error(f'Project "{project}" is not a valid RAPIDA project')
        return
    prj.delete(yes=yes)

@click.command(short_help=f'publish RAPIDA project results to Azure and GeoHub')
@click.option('-p', '--project',
              default=None,
              type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
              help="Optional. A project folder with rapida.json can be specified. If not, current directory is considered as a project folder.")
@click.option('-y', '--yes',
              is_flag=True,
              default=False,
              help="Optional. If True, it will automatically answer yes to prompts. Default is False.")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug"
              )
def publish(project: str, yes: bool = False, debug: bool =False):
    """
    Publish project data to Azure and open GeoHub registration page URL.

    Usage:

        If you are already in a project folder, run the below command:
        rapida publish

        If you are not in a project folder, run the below command:
        rapida publish --project=<project folder path>

        If you answer all prompts to yes, use '--yes' option of the command.
    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    if project is None:
        project = os.getcwd()
    else:
        os.chdir(project)
        logger.info(f"Publish command is executed at the project folder: {project}")
    prj = Project(path=project)
    if not prj.is_valid:
        logger.error(f'Project "{project}" is not a valid RAPIDA project')
        return
    prj.publish(yes=yes)