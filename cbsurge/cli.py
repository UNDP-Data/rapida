
from cbsurge.util import setup_logger
from cbsurge.admin import admin
from cbsurge.exposure.builtenv import builtenv
import click



@click.group
def cli(ctx):
    """Main CLI for the application."""
    pass
cli.add_command(admin)
cli.add_command(builtenv)

from cbsurge.exposure.population import population
from cbsurge.exposure.builtenv import builtenv
#from cbsurge.stats import stats


@click.group


def cli():
    """Main CLI for the application."""
    pass
cli.add_command(admin)
cli.add_command(population)
cli.add_command(builtenv)

#cli.add_command(stats)



if __name__ == '__main__':

    logger = setup_logger('rapida', make_root=True)
    cli()