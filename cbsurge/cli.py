import logging

from cbsurge.util import setup_logger
from cbsurge.admin import admin
from cbsurge.project import create, list, upload, download, publish
from cbsurge.initialize import init
from cbsurge.assess import assess
from cbsurge.stats import stats
import click


@click.group
@click.pass_context
def cli(ctx=None):
    """UNDP Crisis Bureau Rapida tool.

    This command line tool is designed to assess various geospatial variables
    representing exposure and vulnerability aspects of geospatial risk induced
    by natural hazards.
    """

cli.add_command(admin)
cli.add_command(init)
cli.add_command(assess)
cli.add_command(stats)
cli.add_command(create)
cli.add_command(list)
cli.add_command(upload)
cli.add_command(download)
cli.add_command(publish)


if __name__ == '__main__':

    logger = setup_logger('rapida')
    cli()