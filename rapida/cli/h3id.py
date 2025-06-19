import logging
import click
from rapida.util.add_h3id import add_h3id


class LayerType(click.ParamType):
    name = 'ogr_layer_type'

    def convert(self, value, param, ctx):
        try:
            return int(value)
        except ValueError:
            return value  # keep as string


layer_type = LayerType()

logger = logging.getLogger(__name__)

@click.command(short_help=f'add h3id to a vector dataset')
@click.argument('dataset_path', type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, file_okay=True))
@click.option('--layer', '-l', default=0, show_default=True, type=layer_type, help='layer name to add h3id to.')
@click.option('-c', '--h3id_column', default="h3id", show_default=True, type=str, help="column name to add h3id to.")
@click.option('--h3id_precision', '-p', default=7, show_default=True, type=int, help='precision of h3id')
def addh3id(dataset_path=None, h3id_precision=7, layer=None, h3id_column=None):
    add_h3id(dataset_path=dataset_path, h3id_precision=h3id_precision, layer=layer, h3id_column=h3id_column)