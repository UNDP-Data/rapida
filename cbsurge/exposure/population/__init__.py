import asyncio

import click

from cbsurge.exposure.population.worldpop import population_sync, process_aggregates, download


@click.group()
def population():
    f"""Command line interface for {__package__} package"""
    pass


@population.command()
@click.option('--force-reprocessing', help='Force reprocessing of data', is_flag=True)
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--download-path', help='Download data locally', required=False)
def sync(force_reprocessing, country, download_path):
    asyncio.run(population_sync(force_reprocessing=force_reprocessing, country_code=country, download_path=download_path))


@population.command()
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--force-reprocessing', help='Force reprocessing of data', is_flag=True)
@click.option('--download-path', help='Download data locally', required=False)
@click.option('--age-group', help='The age group to process the data for', type=click.Choice(['child', 'active', 'elderly']))
@click.option('--sex', help='Path to the downloaded data', type=click.Choice(['male', 'female']))
def download(country, force_reprocessing, download_path, age_group, sex):
    asyncio.run(download(force_reprocessing=force_reprocessing, country_code=country, download_path=download_path, age_group=age_group, sex=sex))

@population.command()
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--age-group', help='The age group to process the data for', type=click.Choice(['child', 'active', 'elderly']))
@click.option('--sex', help='Path to the downloaded data', type=click.Choice(['male', 'female']))
@click.option('--download-path', help='Download data locally', required=False)
@click.option('--force-reprocessing', help='Force reprocessing of data', is_flag=True)
def run_aggregate(country, age_group, sex, download_path, force_reprocessing):
    asyncio.run(process_aggregates(country_code=country, age_group=age_group, sex=sex, download_path=download_path, force_reprocessing=force_reprocessing))