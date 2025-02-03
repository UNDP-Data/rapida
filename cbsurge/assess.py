import logging
from cbsurge.session import Session
import click
logger = logging.getLogger(__name__)


@click.command()

def assess(component=None, variable=None, **kwargs):
    with Session() as session:
        project = session.project
