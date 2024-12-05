import asyncclick as click
import anyio
from download_worldpop import download_data


@click.command()
@click.option('--force-reprocessing', help='Force reprocessing of data', is_flag=True)
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--download-locally', help='Download data locally', is_flag=True)
async def run_download(force_reprocessing, country, download_locally):
    await download_data(force_reprocessing=force_reprocessing, country_code=country, download_locally=download_locally)




if __name__ == '__main__':
    run_download()
