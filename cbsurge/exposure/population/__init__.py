import asyncio

import click

from cbsurge.exposure.population.worldpop import download_data


@click.group()
def population():
    f"""Command line interface for {__package__} package"""
    pass


@population.command()
@click.option('--force-reprocessing', help='Force reprocessing of data', is_flag=True)
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--download-path', help='Download data locally', required=False)
def run_download(force_reprocessing, country, download_path):
    asyncio.run(download_data(force_reprocessing=force_reprocessing, country_code=country, download_path=download_path))