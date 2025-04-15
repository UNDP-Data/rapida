import os
import click
import logging
from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger
from cbsurge.project.project import Project

logger = logging.getLogger(__name__)



@click.command(short_help=f'publish RAPIDA project results to Azure and GeoHub')
@click.option('-p', '--project',
              default=None,
              type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
              help="Optional. A project folder with rapida.json can be specified. If not, current directory is considered as a project folder.")
@click.option('--no-input',
              is_flag=True,
              default=False,
              help="Optional. If True, it will automatically answer yes to prompts. Default is False.")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug"
              )
def publish(project: str, no_input: bool = False, debug: bool =False):
    """
    Publish project data to Azure and open GeoHub registration page URL.

    `--no-input` option can answer yes to all prompts. Default is False.

    Usage:

        rapida publish: If you are already in a project folder

        rapida publish --project=<project folder path>: If you are not in a project folder

    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    if project is None:
        project = os.getcwd()
    else:
        os.chdir(project)
        click.echo(f"Publish command is executed at the project folder: {project}")
    prj = Project(path=project)
    if not prj.is_valid:
        click.echo(f'Project "{project}" is not a valid RAPIDA project')
        return
    prj.publish(no_input=no_input)