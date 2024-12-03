from cbsurge.admin import admin
import click

@click.group
@click.pass_context
def cli(ctx):
    """Main CLI for the application."""
    pass
cli.add_command(admin)

if __name__ == '__main__':
    cli()