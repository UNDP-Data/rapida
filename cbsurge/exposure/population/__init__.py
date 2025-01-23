import asyncio

import click

from cbsurge.exposure.population.worldpop import population_sync, process_aggregates, run_download


@click.group()
def population():
    f"""Command line interface for {__package__} package"""
    pass


@population.command(no_args_is_help=True)
@click.option('--force-reprocessing', help='Force reprocessing of data even if the data specified already exists', is_flag=True)
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--download-path', help='The local path to save the data to. If not provided, it will automatically save to the provided azure container that was set when initializing with `rapida init`', required=False)
@click.option('--all-data', help='Sync all datasets for all countries. It should not be used together with --country flag', is_flag=True, default=False)
def sync(force_reprocessing, country, download_path, all_data):
    if all_data and country:
        raise click.UsageError("The --country flag should not be used together with --all-data flag")
    asyncio.run(population_sync(force_reprocessing=force_reprocessing, country_code=country, download_path=download_path, all_data=all_data))


@population.command(no_args_is_help=True)
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--download-path', help='The local path to save the data to. If not provided, it will automatically save to the provided azure container that was set when initializing with `rapida init`', required=True)
@click.option('--age-group', help='The age group (child, active or elderly) to process the data for', type=click.Choice(['child', 'active', 'elderly']))
@click.option('--sex', help='The sex (male or female) to process the data for', type=click.Choice(['male', 'female']))
@click.option('--non-aggregates', help='Download only aggregate files', is_flag=True, default=False)
def download(country, download_path, age_group, sex, non_aggregates):
    asyncio.run(run_download(country_code=country, download_path=download_path, age_group=age_group, sex=sex, non_aggregates=non_aggregates))

@population.command(no_args_is_help=True)
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--age-group', help='The age group (child, active or elderly) to process the data for', type=click.Choice(['child', 'active', 'elderly']))
@click.option('--sex', help='The sex (male or female) to process the data for', type=click.Choice(['male', 'female']))
@click.option('--download-path', help='The local path to save the data to. If not provided, it will automatically save to the provided azure container that was set when initializing with `rapida init`', required=False)
@click.option('--force-reprocessing', help='Force reprocessing of data even if the data specified already exists', is_flag=True)
def aggregate(country, age_group, sex, download_path, force_reprocessing):
    asyncio.run(process_aggregates(country_code=country, age_group=age_group, sex=sex, download_path=download_path, force_reprocessing=force_reprocessing))