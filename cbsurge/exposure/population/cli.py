import click
from cbsurge.exposure.population.download_worldpop import download_data


@click.command()
@click.option('--country', help='Country code', type=str)
@click.option('--download_locally', help='Download files locally', type=bool)
@click.option('--force-reprocessing', help='Force redownload', type=bool)
async def download_files(country=None, download_locally=False, force_reprocessing=True):
    """
    Download files for a given country.
    Args:
        force_reprocessing: (bool) Force reprocessing of any files found in azure
        country: (str) Country code
        download_locally: (bool) Download files locally
    Returns: None
    """
    return await download_data(country_code=country, download_locally=download_locally, force_reprocessing=force_reprocessing)

