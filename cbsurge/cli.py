from cbsurge.admin import admin
from cbsurge.exposure.builtenv import builtenv
import click

@click.group
def cli(ctx):
    """Main CLI for the application."""
    pass
cli.add_command(admin)
cli.add_command(builtenv)


if __name__ == '__main__':
    cli()