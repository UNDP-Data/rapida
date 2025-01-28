import click
from cbsurge.util import BboxParamType

@click.command(no_args_is_help=True)
@click.option('-b', '--bbox',
              required=False,
              type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north'
              )
@click.option('-c', '--country',
              required=False,
              multiple=True,
              type = str,
              help='The list of target country ISO3 codes'
              )
@click.option('-a', '--admin',
              required=False,
              type = str,
              help='Admin data file path.'
              )
@click.option('-m', '--component',
              required=False,
              type = str,
              help='The list of component names. If skipped, will assess all available components,'
              )
def assess():
    """
    This command do assessment for user interested area to download component variables and make zonal statistics for further assessment.
    """
    pass