import click
from cbsurge.az.fileshare import list_projects

from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger


logger = setup_logger()


@click.command(short_help=f'list RAPIDA projects/folders located in default Azure file share')
def list():
    if not is_rapida_initialized():
        return

    const = '-'*15
    tabs = '\t'*1
    click.echo(f'{const} Available RAPIDA projects {const}')
    for project_name in list_projects():
        click.echo(f'{tabs}"{project_name}"')