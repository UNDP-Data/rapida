
import os
import click
import logging
from osgeo import gdal
import sys
from cbsurge.az.fileshare import list_projects, upload_project, download_project
from rich.progress import Progress
from cbsurge.util.setup_logger import setup_logger
from cbsurge.project.project import Project


logger = logging.getLogger(__name__)
gdal.UseExceptions()

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

    logger = logging.getLogger('rapida')
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
    const = '-'*15
    tabs = '\t'*1
    click.echo(f'{const} Available RAPIDA projects {const}')
    for project_name in list_projects():
        click.echo(f'{tabs}"{project_name}"')




@click.command(short_help=f'upload a RAPIDA project to Azure file share')

@click.argument('project_folder', nargs=1,
                type=click.Path(exists=True, dir_okay=True, readable=True, resolve_path=True) )
@click.option('--max_concurrency', default=4, show_default=True, type=int,
              help=f'The number of threads to use when uploading a file')
@click.option('--overwrite','-o',is_flag=True,default=False,
              help="Whether to overwrite the project in case it already exists."
)

def upload(project_folder=None,max_concurrency=None,overwrite=None):

    project_folder = os.path.abspath(project_folder)
    assert os.path.exists(project_folder), f'{project_folder} does not exist'


    with Progress() as progress:
        progress.console.print(f'Going to upload {project_folder} to Azure')
        upload_project(project_folder=project_folder, progress=progress, overwrite=overwrite, max_concurrency=max_concurrency)
        progress.console.print(f'Rapida project "{project_folder}" was uploaded successfully to Azure')

@click.command(short_help=f'download a RAPIDA project from Azure file share')

@click.argument('name', nargs=1 )
@click.argument('destination_path', type=click.Path(exists=True, dir_okay=True, readable=True, resolve_path=True))

@click.option('--max_concurrency', default=4, show_default=True, type=int,
              help=f'The number of threads to use when downloading a file')
@click.option('--overwrite','-o',is_flag=True,default=False, help="Whether to overwrite the project in case it already exists locally."
)

def download(name=None, destination_path=None, max_concurrency=None,overwrite=None ):

    with Progress() as progress:
        progress.console.print(f'Going to download rapida project "{name}" from Azure')
        download_project(name=name, dst_folder=destination_path, progress=progress, overwrite=overwrite, max_concurrency=max_concurrency)
        progress.console.print(f'Project "{name}" was downloaded successfully to {os.path.join(destination_path, name)}')










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

    if project is None:
        project = os.getcwd()
    else:
        os.chdir(project)
        logger.info(f"Publish command is executed at the project folder: {project}")
    prj = Project(path=project)
    prj.publish(yes=yes)