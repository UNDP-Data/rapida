import os
import click
import logging
from rich.progress import Progress
from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger
from cbsurge.project.project import Project


logger = logging.getLogger(__name__)


@click.command(short_help=f'upload a RAPIDA project to Azure file share')
@click.option('-p', '--project',
              default=None,
              type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
              help="Optional. A project folder with rapida.json can be specified. If not, current directory is considered as a project folder.")
@click.option('--max_concurrency', '-c',
              default=4,
              show_default=True,
              type=int,
              help=f'The number of threads to use when uploading a file')
@click.option('--force','-f',
              is_flag=True,
              default=False,
              help="Whether to overwrite the project in case it already exists.")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug")
def upload(project=None, max_concurrency=4, force=False, debug: bool =False):
    """
    Upload an entire project folder to Azure File Share

    Usage:

        rapida upload: If you are already in a project folder

        rapida upload --project=<project folder path>: If you are not in a project folder

    To use `-f/--force`, project data will be overwritten if it already exists in Azure File Share..
    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    if project is None:
        project = os.getcwd()
    else:
        os.chdir(project)
        click.echo(f"Upload command is executed at the project folder: {project}")

    prj = Project(path=project)

    if not prj.is_valid:
        click.echo(f'Project "{project}" is not a valid RAPIDA project')
        return

    if click.confirm('Are you sure uploading this project to Azure? Y/Yes to continue, Enter/No to cancel', abort=True):
        click.echo(f'Going to upload {project} to Azure')
        with Progress() as progress:
            prj.upload(progress=progress, overwrite=force, max_concurrency=max_concurrency)
        click.echo(f'Rapida project "{project}" was uploaded successfully to Azure')
    else:
        click.echo('Upload was cancelled.')