import logging

from cbsurge.util.setup_logger import setup_logger
from cbsurge.admin import admin
from cbsurge.project import create, list, upload, download, publish, delete
from cbsurge.initialize import init
from cbsurge.assess import assess
import click


@click.group
@click.pass_context
def cli(ctx):
    """UNDP Crisis Bureau Rapida tool.

    This command line tool is designed to assess various geospatial variables
    representing exposure and vulnerability aspects of geospatial risk induced
    by natural hazards.
    """
    logger = setup_logger(name='rapida', make_root=False)


cli.add_command(admin)
cli.add_command(init)
cli.add_command(assess)
cli.add_command(create)
cli.add_command(list)
cli.add_command(upload)
cli.add_command(download)
cli.add_command(delete)
cli.add_command(publish)


if __name__ == '__main__':


    cli()