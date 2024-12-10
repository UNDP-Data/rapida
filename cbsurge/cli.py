import asyncclick as click

from cbsurge.admin import admin
from cbsurge.exposure.population import population


@click.group
@click.pass_context
def cli(ctx):
    """Main CLI for the application."""
    pass
cli.add_command(admin)
cli.add_command(population)


if __name__ == '__main__':
    cli()