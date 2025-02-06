import asyncio
import os
from typing import List
import logging
import pycountry
from osgeo import gdal
from cbsurge.components.component_base import ComponentBase
from cbsurge.project import Project
from cbsurge.util import get_geographic_bbox
import click
from cbsurge import isobbox
from cbsurge.components.population.worldpop import population_sync, process_aggregates, run_download

COUNTRY_CODES = set([c.alpha_3 for c in pycountry.countries])
logger = logging.getLogger(__name__)

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
    """
    Download population data (sex and age structures) from worldpop,
    then do additional processing to convert them to Cloud Optimised GeoTiff and aggregate them for our needs.

    If `--download-path` is provided, it will save the data to a local folder. Otherwise, it will automatically save to the provided azure container.

    You can process for a specific country by using `--country` option or all countries with `--all-data` option.

    Use `--force-reprocessing` parameter, it will force reprocessing of data even if the data specified already exists.

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
    asyncio.run(population_sync(force_reprocessing=force_reprocessing, country_code=country, download_path=download_path, all_data=all_data))


@population.command(no_args_is_help=True)
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
def download(country, download_path, age_group, sex, non_aggregates):
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

    Original worldpop populatin data is split into each age group. If you would like to download original worldpop data, please use `--non-aggregates` option

        rapida population download --country={COUNTRY ISO 3 code} --download-path={DOWNLOAD_PATH} --non-aggregates

    If you would like to download original data for specific conditions, use --sex or --age-group together. `--age-group=total` is not supported for non-aggregate data.
    """
    asyncio.run(run_download(country_code=country, download_path=download_path, age_group=age_group, sex=sex, non_aggregates=non_aggregates))

@population.command(no_args_is_help=True)
@click.option('--country', help='The ISO3 code of the country to process the data for')
@click.option('--age-group', help='The age group (child, active or elderly) to process the data for', type=click.Choice(['child', 'active', 'elderly']))
@click.option('--sex', help='The sex (male or female) to process the data for', type=click.Choice(['male', 'female']))
@click.option('--download-path', help='The local path to save the data to. If not provided, it will automatically save to the provided azure container that was set when initializing with `rapida init`', required=False)
@click.option('--force-reprocessing', help='Force reprocessing of data even if the data specified already exists', is_flag=True)
def aggregate(country, age_group, sex, download_path, force_reprocessing):
    """
    Aggregate original worldpop data and aggregate them

    If ` --download-path` is provided, it will save aggregated data to a local folder. Otherwise it will automatically save to the provided azure container.

    Use `--country` to specify a country ISO3 code to process it.

    You can only aggregate for a specific age group or sex by using `--age-group` or `--sex` options.
    """
    asyncio.run(process_aggregates(country_code=country, age_group=age_group, sex=sex, download_path=download_path, force_reprocessing=force_reprocessing))




class PopulationComponent(ComponentBase):

    def __init__(self, year=2020, country:str = None):


        self.year = year
        # self.country = country
        # if self.country is None:
        project = Project(path=os.getcwd())
        print(project.geopackage_file_path)
        with gdal.OpenEx(project.geopackage_file_path, gdal.OF_READONLY|gdal.OF_VECTOR) as vds:
            layer = vds.GetLayerByName('polygons')
            geo_bbox = get_geographic_bbox(layer=layer)
            self.countries = isobbox.bbox2iso31(*geo_bbox)

        print(self.countries)

        super().__init__()

    def assess(self, variables: List[str] = None, **kwargs) -> str:
        logger.info(f'Assessing "{self.component_name}" component {variables}')

        vrs = list(filter(lambda v: v in self.variable_names, variables or self.variable_names))
