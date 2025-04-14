import logging
import click
from cbsurge.az.fileshare import list_projects
from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger


logger = logging.getLogger(__name__)


@click.command(short_help=f'list RAPIDA projects/folders located in default Azure file share')
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug"
              )
def list(debug=False):
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    const = '-'*15
    tabs = '\t'*1
    click.echo(f'{const} Available RAPIDA projects {const}')
    for project_name in list_projects():
        click.echo(f'{tabs}"{project_name}"')