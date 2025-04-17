import os
import click
import logging
import pyogrio
from cbsurge.session import is_rapida_initialized
from cbsurge.util.setup_logger import setup_logger
from cbsurge.project.project import Project

logger = logging.getLogger(__name__)


def validate_layers(ctx, param, value):
    """
    click callback function to validate -l/--layer value.
    This function checks whether user inputted layer names are matched to actual layer name in project GeoPackage.
    """
    if not value:
        return value

    project_path = ctx.params.get('project')
    if project_path is None:
        project_path = os.getcwd()
    prj = Project(path=project_path)

    if not prj.is_valid:
        return value

    gpkg_path = prj.geopackage_file_path

    layers = pyogrio.list_layers(gpkg_path)
    layer_names = layers[:, 0]

    invalid_layers = []
    for layer in value:
        if not layer in layer_names:
            invalid_layers.append(layer)

    if len(invalid_layers) > 0:
        raise click.BadParameter(f"Invalid layer{'s' if len(invalid_layers) > 1 else ''}: {', '.join(invalid_layers)}. Valid options: {", ".join(layer_names)}")

    return value


@click.command(short_help=f'publish RAPIDA project results to Azure and GeoHub')
@click.option('-l', '--layer',
              required=False, multiple=True,
              default=None,
              type=str,
              callback=validate_layers,
              help="Optional. Layer name to publish results to. If this option is not used, the command will convert all layers starting with stats to cloud optimized format.")
@click.option('-p', '--project',
              default=None,
              type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
              help="Optional. A project folder with rapida.json can be specified. If not, current directory is considered as a project folder.")
@click.option('--no-input',
              is_flag=True,
              default=False,
              help="Optional. If True, it will automatically answer yes to prompts. Default is False.")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug"
              )
def publish(project: str, no_input: bool = False, debug: bool =False, layer=None):
    """
    Publish project data to Azure and open GeoHub registration page URL.

    As default, all layers starting with 'stats.' will be converted to cloud optimized format, and uploaded to Azure Blob Container.

    Using `-l/--layer` option to select layers which you want to publish.

    `--no-input` option can answer yes to all prompts. Default is False.

    Usage:

        rapida publish: If you are already in a project folder

        rapida publish --project=<project folder path>: If you are not in a project folder

        rapida publish -l stats.elegrid -l elegrid: publish only two layers from geopackage

    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    if project is None:
        project = os.getcwd()
    else:
        os.chdir(project)
        click.echo(f"Publish command is executed at the project folder: {project}")
    prj = Project(path=project)
    if not prj.is_valid:
        click.echo(f'Project "{project}" is not a valid RAPIDA project')
        return

    target_layers = layer
    if target_layers is None or len(target_layers) == 0:
        gpkg_path = prj.geopackage_file_path
        layers = pyogrio.list_layers(gpkg_path)
        layer_names = layers[:, 0]
        stats_layers = [name for name in layer_names if name.startswith("stats.")]
        has_stats_layers = len(stats_layers) > 0
        if not has_stats_layers:
            click.echo(f"Could not find assessed statistics layer in the project. Please do assess command first for at least one component.")
            return
        target_layers = stats_layers

    prj.publish(no_input=no_input, target_layers=target_layers)