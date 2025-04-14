import os
import click
import logging
from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger
from cbsurge.project.project import Project

logger = logging.getLogger(__name__)



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