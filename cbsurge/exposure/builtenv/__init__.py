import click
from cbsurge.exposure.builtenv.buildings import buildings
@click.group()
def builtenv():
    f"""Command line interface for {__package__} package"""
    pass

builtenv.add_command(buildings)