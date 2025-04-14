from cbsurge.util.setup_logger import setup_logger
from cbsurge.cli.admin import admin
from cbsurge.cli.auth import auth
from cbsurge.cli.init import init
from cbsurge.cli.assess import assess
from cbsurge.cli.create import create
from cbsurge.cli.delete import delete
from cbsurge.cli.list import list
from cbsurge.cli.upload import upload
from cbsurge.cli.download import download
from cbsurge.cli.publish import publish
import click


@click.group
@click.pass_context
def cli(ctx):
    """UNDP Crisis Bureau Rapida tool.

    This command line tool is designed to assess various geospatial variables
    representing exposure and vulnerability aspects of geospatial risk induced
    by natural hazards.
    """
    logger = setup_logger(name='rapida', make_root=False)


cli.add_command(admin)
cli.add_command(init)
cli.add_command(assess)
cli.add_command(auth)
cli.add_command(create)
cli.add_command(list)
cli.add_command(upload)
cli.add_command(download)
cli.add_command(delete)
cli.add_command(publish)


if __name__ == '__main__':


    cli()