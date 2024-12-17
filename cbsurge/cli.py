import click as click
from cbsurge.session import init
from cbsurge.admin import admin
from cbsurge.exposure.builtenv import builtenv
from cbsurge.exposure.population import population
from cbsurge.stats import stats
import click

@click.group

def cli():
    """Main CLI for the application."""
    pass

cli.add_command(admin)
cli.add_command(builtenv)
cli.add_command(init)
cli.add_command(population)
cli.add_command(stats)



if __name__ == '__main__':
    cli()