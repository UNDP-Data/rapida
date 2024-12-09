import asyncclick as click

from upload import upload_files
from download_worldpop import download_data

@click.group
@click.pass_context
async def cli(ctx):
    """Main CLI for the application."""
    pass


@click.command()
@click.option('--force-reprocessing', help='Force reprocessing of data', is_flag=True)
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--download-path', help='Download data locally', required=False)
async def run_download(force_reprocessing, country, download_path):
    await download_data(force_reprocessing=force_reprocessing, country_code=country, download_path=download_path)


@click.command()
@click.option('--local-directory', help='The directory to upload to Azure Blob Storage', required=True)
@click.option('--azure-directory', help='The directory in Azure Blob Storage to upload to', required=True)
async def run_upload(local_directory, azure_directory):
    await upload_files(local_directory=local_directory, azure_directory=azure_directory)



cli.add_command(run_download)
cli.add_command(run_upload)

if __name__ == '__main__':
    cli()