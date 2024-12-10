import asyncclick as click

from cbsurge.exposure.population.worldpop import download_data


@click.group()
async def population():
    f"""Command line interface for {__package__} package"""
    pass


@population.command()
@click.option('--force-reprocessing', help='Force reprocessing of data', is_flag=True)
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--download-path', help='Download data locally', required=False)
async def run_download(force_reprocessing, country, download_path):
    await download_data(force_reprocessing=force_reprocessing, country_code=country, download_path=download_path)