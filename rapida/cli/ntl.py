import logging
import click
from rapida.cli import RapidaCommandGroup

logger = logging.getLogger(__name__)

@click.group(cls=RapidaCommandGroup)
def ntl():
    """Nighttime Lights VIIRS data and impact detection"""
    pass

@ntl.command(short_help=f'Search for available NTL data products across tiers and streams')
async def search():
    logger.info('searching for NTL data')

@ntl.command(short_help=f'Download selected NTL data')
async def download():
    logger.info('Downloading NTL')

@ntl.command(short_help=f'Execute crisis impact detection (48h Alerts / 72h Assessments)')
async def detect():
    logger.info('Detecting impact on the ground')


@ntl.command(short_help=f'Track long-term resilience and recovery curves (2-3 Week horizon)')
async def monitor():
    logger.info('Monitoring recovery')


