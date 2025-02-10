import asyncio
import io
import os
from typing import List
import logging
import pycountry
from pyogrio import write_dataframe, read_info
from cbsurge.core import Component
from cbsurge.project import Project
from cbsurge.session import Session
from cbsurge.core import Variable
from cbsurge.az import blobstorage
from osgeo_utils.gdal_calc import Calc
import click
from cbsurge.components.population.worldpop import population_sync, process_aggregates, run_download
from cbsurge.stats.zst import zonal_stats, sumup
import geopandas
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



class PopulationComponent(Component):

    def __init__(self, year=2020):

        self.year = year
        super().__init__()

    def __call__(self, variables: List[str] = None, **kwargs) -> str:

        logger.info(f'Assessing component "{self.component_name}" ')
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if not var_name in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return


        progress = kwargs.get('progress', None)
        project = Project(path=os.getcwd())

        with Session() as ses:
            variables_data = ses.get_component(self.component_name)
            nvars = len(variables)
            for country in project.countries:
                if progress:
                    variable_task = progress.add_task(
                        description=f'[red]Going to process {nvars} variables', total=nvars)
                for var_name in variables:
                    var_data = variables_data[var_name]
                    v = PopulationVariable(name=var_name,
                                           component=self.component_name,

                                           **var_data)
                    v(year=self.year, country=country, **kwargs)
                    if variable_task and progress:
                        progress.update(variable_task, advance=1, description=f'Assessing {var_name} in {country}')

                if progress and variable_task:progress.remove_task(variable_task)






class PopulationVariable(Variable):

    def compute(self, **kwargs):

        logger.debug(f'Computing {self.name}')
        overwrite = kwargs.get('force_compute', None)

        if not self.dep_vars:
            logger.debug(f'Going to download {self.name} from source files')

            # interpolate templates
            source_blobs = list()
            for source_template in self.sources:
                source_file_path = self.interpolate_template(template=source_template, **kwargs)
                source_blobs.append(source_file_path)
            if not os.path.exists(self._source_folder_):
                os.makedirs(self._source_folder_)
            downloaded_files = asyncio.run(
                blobstorage.download_blobs(src_blobs=source_blobs, dst_folder=self._source_folder_,
                                           progress=kwargs.get('progress', None))
            )
            assert len(self.sources) == len(downloaded_files), f'Not all sources were downloaded for {self.name} variable'
            src_path = self.interpolate_template(template=self.source, **kwargs)
            _, file_name = os.path.split(src_path)
            local_path = os.path.join(self._source_folder_, file_name)
            logger.debug(f'Going to compute {self.name} from {len(downloaded_files)} source files')
            computed_file = sumup(src_rasters=downloaded_files,dst_raster=local_path, overwrite=overwrite)
            assert os.path.exists(computed_file), f'The computed file: {computed_file} does not exists'
            self.local_path = computed_file
        else:
            logger.debug(f'Going to compute {self.name}={self.sources}')
            src_path = self.interpolate_template(template=self.source, **kwargs)
            _, file_name = os.path.split(src_path)
            computed_file = os.path.join(self._source_folder_, file_name)
            sources = self.resolve(**kwargs)
            creation_options = 'TILED=YES COMPRESS=ZSTD BIGTIFF=IF_SAFER BLOCKXSIZE=256 BLOCKYSIZE=256 PREDICTOR=2'
            print(sources)

            ds = Calc(calc=self.sources, outfile=computed_file, projectionCheck=True, format='GTiff',
                      creation_options=creation_options.split(' '), quiet=False, overwrite=overwrite, **sources)

            assert os.path.exists(computed_file), f'The computed file: {computed_file} does not exists'
            self.local_path = computed_file

    def resolve(self, evaluate=False, **kwargs):
        with Session() as s:
            # project = Project(path=os.getcwd())
            sources = dict()
            for var_name in self.dep_vars:
                var_dict = s.get_variable(component=self.component, variable=var_name)
                var = self.__class__(name=var_name, component=self.component, **var_dict)
                var_local_path = var(**kwargs) # assess
                if evaluate:
                    var.evaluate(**kwargs)
                sources[var_name] = var_local_path or var.local_path
            return sources


    def evaluate(self, **kwargs):



        dst_layer = f'stats.{self.component}'
        with Session() as s:
            project = Project(path=os.getcwd())
            layers = geopandas.list_layers(project.geopackage_file_path)
            lnames = layers.name.tolist()
            if dst_layer in lnames:
                polygons_layer = dst_layer
            else:
                polygons_layer='polygons'
            if self.operator:
                assert os.path.exists(self.local_path), f'{self.local_path} does not exist'
                logger.debug(f'Evaluating variable {self.name} using zonal stats')
                # raster variable, run zonal stats
                gdf = zonal_stats(src_rasters=[self.local_path],
                                      polygon_ds=project.geopackage_file_path,
                                      polygon_layer=polygons_layer, vars_ops=[(self.name, self.operator)]
                                      )
            else:
                # we eval inside GeoDataFrame
                logger.debug(f'Evaluating variable {self.name} using GeoPandas eval')
                gdf = geopandas.read_file(filename=project.geopackage_file_path,layer=dst_layer)
                expr = f'{self.name}={self.sources}'
                logger.debug(expr)
                gdf.eval(expr, inplace=True)

            # merge into stats layer for component
            # Here we rely on OGR append flag. With GPKG format and because zonal_stats


            write_dataframe(
                df=gdf,path=project.geopackage_file_path, layer=dst_layer, overwrite=True,
            )

            # print(read_info(project.geopackage_file_path,layer=dst_layer)['fields'].tolist())

