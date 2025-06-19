from rapida.util.setup_logger import setup_logger
from rapida.cli.admin import admin
from rapida.cli.auth import auth
from rapida.cli.init import init
from rapida.cli.assess import assess
from rapida.cli.create import create
from rapida.cli.delete import delete
from rapida.cli.list import list
from rapida.cli.upload import upload
from rapida.cli.download import download
from rapida.cli.publish import publish
from rapida.cli.h3id import addh3id


import click
import nest_asyncio
nest_asyncio.apply()



class RapidaCommandGroup(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=RapidaCommandGroup, context_settings=dict(help_option_names=['-h', '--help']))
@click.pass_context
def cli(ctx):
    """UNDP Crisis Bureau Rapida tool.

    This command line tool is designed to assess various geospatial variables
    representing exposure and vulnerability aspects of geospatial risk induced
    by natural hazards.
    """
    logger = setup_logger(name='rapida', make_root=False)

cli.add_command(init)
cli.add_command(auth)
cli.add_command(admin)
cli.add_command(create)
cli.add_command(assess)
cli.add_command(list)
cli.add_command(download)
cli.add_command(upload)
cli.add_command(publish)
cli.add_command(delete)
cli.add_command(addh3id)


if __name__ == '__main__':
    cli()