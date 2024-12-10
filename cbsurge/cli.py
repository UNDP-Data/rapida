from cbsurge.admin import admin
import click
@click.group

def cli():
    """Main CLI for the application."""
    pass
cli.add_command(admin)


if __name__ == '__main__':
    cli()