import os
import click
import logging
from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger
from cbsurge.project.project import Project

logger = setup_logger()


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