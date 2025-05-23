import os
import click
import logging
from rapida.session import is_rapida_initialized
from rapida.util.setup_logger import setup_logger
from rapida.project.project import Project

logger = logging.getLogger(__name__)



@click.command(short_help=f'delete a RAPIDA project from Azure file share')
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
              help="Set log level to debug")
def delete(project: str, no_input: bool = False, debug: bool =False):
    """
    Delete a project from local storage and Azure File Share if it was uploaded.

    `--no-input` option can answer yes to all prompts. Default is False.

    Note. Please be careful that it will delete all projects data from Azure and your local storage if you use `--no-input`.

    Usage:

        rapida delete: If you are already in a project folder

        rapida delete --project=<project folder path>: If you are not in a project folder

    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    if project is None:
        project = os.getcwd()
    else:
        os.chdir(project)
        click.echo(f"Delete command is executed at the project folder: {project}")

    prj = Project(path=project)
    if not prj.is_valid:
        click.echo(f'Project "{project}" is not a valid RAPIDA project')
        return
    prj.delete(no_input=no_input)