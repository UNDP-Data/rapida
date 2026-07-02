
import click
from aenum import no_arg

from rapida.cli import RapidaCommandGroup
from rapida.components.population import run_download, process_aggregates, population_sync
from rapida.components.population.wpopstac import stac_search
from rapida.components.population import constants as wpopconst
from rapida.util.bbox_param_type import BboxParamType
import tempfile

@click.group(cls=RapidaCommandGroup, no_args_is_help=True)
def population():
    """
    Population data management commands.
    Returns
    -------

    """
    pass


@population.command(short_help="Synchronize population data from world pop with data stored in our Azure Storage")
@click.option('--force-reprocessing', help='Force reprocessing of data even if the data specified already exists', is_flag=True)
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--download-path', help='The local path to save the data to. If not provided, it will automatically save to the provided azure container that was set when initializing with `rapida init`', required=False)
@click.option('--year', help='target year', type=int, default=2020)
@click.option('--all-data', help='Sync all datasets for all countries. It should not be used together with --country flag', is_flag=True, default=False)
async def sync(force_reprocessing, country, download_path, year, all_data):
    """
    Download population data (sex and age structures) from worldpop,
    then do additional processing to convert them to Cloud Optimised GeoTiff and aggregate them for our needs.

    If `--download-path` is provided, it will save the data to a local folder. Otherwise, it will automatically save it to the provided azure container.

    You can process for a specific country by using the ` -- country ` option or all countries with `--all-data` option.

    Use the ` -- force-reprocessing ` parameter, it will force reprocessing of data even if the data specified already exists.

    Usage example:

    - Download for a country to a local folder

        rapida population sync --country=RWA --download-path=/data

    - Download for all countries to a local folder

        rapida population sync --all-data --download-path=/data

    - Download and upload to Azure Blob Storage container for a country

        rapida population sync --country=RWA
    """
    if all_data and country:
        raise click.UsageError("The --country flag should not be used together with --all-data flag")
    await population_sync(force_reprocessing=force_reprocessing, country_code=country, download_path=download_path, all_data=all_data, year=year)


@population.command(short_help="Download population data (Cloud Optimised GeoTiff format) from UNDP Azure Blob Storage.")
@click.option('--country',
              help='The ISO3 code of the country to process the data for')
@click.option('--download-path',
              help='The local path to save the data to. If not provided, it will automatically save to the provided azure container that was set when initializing with `rapida init`', required=True)
@click.option('--age-group',
              help='The age group (child, active, elderly or total) to process the data for. total is not supported with non aggregate option',
              type=click.Choice(['child', 'active', 'elderly', 'total']))
@click.option('--sex',
              help='The sex (male or female) to process the data for',
              type=click.Choice(['male', 'female']))
@click.option('--non-aggregates',
              help='Download only non aggregated files (original worldpop data). Default is to download only aggregated data.',
              is_flag=True,
              default=False)
async def download(country, download_path, age_group, sex, non_aggregates):
    """
    Download population data (Cloud Optimised GeoTiff format) from UNDP Azure Blob Storage.

    Usage:
    To simply download all aggregated worldpop population data for a country, execute the command like below.

        rapida population download --country={COUNTRY ISO 3 code} --download-path={DOWNLOAD_PATH}

    Check country ISO code from https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes.

    It downloads all data (age group (total, child, active, elderly), and sex (male, female) according to given options to the command.

    If only specific sex data needs to be downloaded, use `--sex` like below.
        rapida population download --country={COUNTRY ISO 3 code} --sex={female|male} --download-path={DOWNLOAD_PATH}


    If only specific age data needs to be downloaded, use `--age-group` like below.

        rapida population download --country={COUNTRY ISO 3 code} --age-group={child|active|elderly|total} --download-path={DOWNLOAD_PATH}

    if only total data needs to be downloaded, use `--age-group=total` like below.

        rapida population download --country={COUNTRY ISO 3 code} --age-group=total --download-path={DOWNLOAD_PATH}

    Original worldpop population data is split into each age group. If you would like to download original worldpop data, please use `--non-aggregates` option

        rapida population download --country={COUNTRY ISO 3 code} --download-path={DOWNLOAD_PATH} --non-aggregates

    If you would like to download original data for specific conditions, use --sex or --age-group together. `--age-group=total` is not supported for non-aggregate data.
    """
    await run_download(country_code=country, download_path=download_path, age_group=age_group, sex=sex, non_aggregates=non_aggregates)

@population.command(short_help="Download original worldpop data and aggregate them")
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--age-group', help='The age group (child, active or elderly) to process the data for', type=click.Choice(['child', 'active', 'elderly']))
@click.option('--sex', help='The sex (male or female) to process the data for', type=click.Choice(['male', 'female']))
@click.option('--download-path', help='The local path to save the data to. If not provided, it will automatically save to the provided azure container that was set when initializing with `rapida init`', required=False)
@click.option('--force-reprocessing', help='Force reprocessing of data even if the data specified already exists', is_flag=True)
async def aggregate(country, age_group, sex, download_path, force_reprocessing):
    """
    Download original worldpop data and aggregate them

    If ` --download-path` is provided, it will save aggregated data to a local folder. Otherwise it will automatically save to the provided azure container.

    Use `--country` to specify a country ISO3 code to process it.

    You can only aggregate for a specific age group or sex by using `--age-group` or `--sex` options.
    """
    await process_aggregates(country_code=country, age_group=age_group, sex=sex, download_path=download_path, force_reprocessing=force_reprocessing)





@population.command( short_help=f'Search world population data from STAC server')

@click.option('-b', '--bbox',
              required=True,
              type=BboxParamType(),
              help='Bounding box xmin/west, ymin/south, xmax/east, ymax/north'
              )


@click.option("--year", "year",
              type=int,
              required=False,
              help='Year for which to search population'
              )
@click.option("--sex-category", "sex_category",
                type=click.Choice(list(wpopconst.SEX_MAPPING.values()) + ['total'], case_sensitive=False),
                required=True,
                default='total',
                help=f'Aggregate population based on sex category'
    )

@click.option( "--age-group", "age_group",
                type=click.Choice(list(wpopconst.WORLDPOP_AGE_MAPPING) + ['total'], case_sensitive=False),
                required=True,
                default='total',
                help=f'Sex of the population'
    )

# @click.option(
#     "--dst-dir",
#     "dst_dir",     # Function argument name
#     type=click.Path(
#         exists=False,      # Set to True if you want Click to fail if the dir doesn't exist yet
#         file_okay=False,   # Strictly enforce that this is a directory, not a file
#         dir_okay=True,
#         resolve_path=True  # Resolves relative paths (like '.') to absolute paths automatically
#     ),
#     default=tempfile.gettempdir(),           # Defaults to the current working directory
#     show_default=True,     # Tells the user what the default is in the --help menu
#     help="Destination directory to save the downloaded the images."
# )




# @click.option(
#     '--cmask', '-cm', "mask_clouds",
#     is_flag=True,
#     help=(
#             "Enable strict Cloud Masking (ignores pixels with NASA quality flags of 3). "
#             "Disable this flag during major storm events to prevent atmospheric noise "
#             "from erroneously masking out legitimate blackout signals."
#     ),
#     default=False
# )



@click.pass_context
def search(ctx, bbox:tuple[float, float, float, float]=None, year:int=None, sex_category:str=None, age_group:str=None):
    progress = ctx.obj.get('progress')
    with progress:
        urls = stac_search(bbox=bbox, year=year, sex_category=sex_category,age_group=age_group)






