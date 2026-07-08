from rapida.cli.aclick import RapidaCommandGroup
from rapida.cli.population import population

from rapida.cli.admin import admin
from rapida.cli.auth import auth
from rapida.cli.init import init
from rapida.cli.assess import assess
from rapida.cli.create import create
from rapida.cli.delete import delete
from rapida.cli.list import list_project
from rapida.cli.upload import upload
from rapida.cli.download import download
from rapida.cli.publish import publish
from rapida.cli.h3id import addh3id
from rapida.cli.ntl import ntl
from rapida.cli.connectivity import connectivity
from rich.progress import Progress
import click
import nest_asyncio
nest_asyncio.apply()
#import uvloop
# import asyncio
#
# asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
#





@click.group(cls=RapidaCommandGroup, )

@click.pass_context
def cli(ctx):
    """UNDP Crisis Bureau Rapida tool.

    This command line tool is designed to assess various geospatial variables
    representing exposure and vulnerability aspects of geospatial risk induced
    by natural hazards.
    """


    # 2. Ensure ctx.obj is initialized as a container dictionary
    ctx.ensure_object(dict)

    # 3. Instantiate a beautifully configured rich Progress engine
    # progress = Progress(
    #     TextColumn("[progress.description]{task.description}"),
    #     BarColumn(bar_width=40),
    #     TaskProgressColumn(),
    #     MofNCompleteColumn(),  # e.g., "3/10 granules"
    #     TimeRemainingColumn(),
    #     transient=True  # Clean up bars from terminal upon completion
    # )
    progress = Progress(disable=False, console=None, transient=True)

    # 4. Spin up the progress display canvas
    #progress.start()

    # 5. Inject it into Click's shared context object
    ctx.obj['progress'] = progress

    # 6. Critical Safeguard: Register a teardown callback to exit the progress frame cleanly
    ctx.call_on_close(lambda: progress.stop())

cli.add_command(init)
cli.add_command(auth)
cli.add_command(admin)
cli.add_command(create)
cli.add_command(assess)
cli.add_command(list_project)
cli.add_command(download)
cli.add_command(upload)
cli.add_command(publish)
cli.add_command(delete)
cli.add_command(addh3id)
cli.add_command(population)
cli.add_command(ntl)
cli.add_command(connectivity)

if __name__ == '__main__':
    cli()