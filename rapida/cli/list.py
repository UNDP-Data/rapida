import logging
import click
from rapida.az.fileshare import list_projects
from rapida.session import is_rapida_initialized
from rapida.util.setup_logger import setup_logger


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

    project_names = []
    for project_name in list_projects():
        project_names.append(project_name)

    if len(project_names) == 0:
        click.echo("No projects found.")
        return

    header = "Project"
    max_len = max(len(name) for name in project_names)
    line_len = max(len(header), max_len, 10)

    click.echo(header)
    click.echo('-' * line_len)
    for project_name in project_names:
        click.echo(project_name)
