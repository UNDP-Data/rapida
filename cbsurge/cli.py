
from cbsurge.util import setup_logger
from cbsurge.admin import admin
from cbsurge.project import create
from cbsurge.initialize import init
from cbsurge.exposure.builtenv import builtenv
from cbsurge.exposure.population import population
from cbsurge.assess import assess
from cbsurge.stats import stats
import click


@click.group
def cli():
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


if __name__ == '__main__':

    logger = setup_logger('rapida', make_root=True)
    cli()