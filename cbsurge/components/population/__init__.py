import asyncio
import io
import os
from typing import List
import logging
import pycountry
from pyogrio import write_dataframe, read_info
from cbsurge.core.component import Component
from cbsurge.project import Project
from cbsurge.session import Session
from cbsurge.core.variable import Variable
from cbsurge.az import blobstorage
from osgeo_utils.gdal_calc import Calc
from osgeo import gdal, osr
import click
from cbsurge.components.population.worldpop import population_sync, process_aggregates, run_download
from cbsurge.stats.zst import zonal_stats, sumup, zst
import geopandas
from cbsurge.components.population.pop_coefficient import get_pop_coeff
from cbsurge.util.proj_are_equal import proj_are_equal



COUNTRY_CODES = set([c.alpha_3 for c in pycountry.countries])
logger = logging.getLogger('rapida')
gdal.UseExceptions()
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

    base_year =  2020

    def __call__(self, variables: List[str] = None, **kwargs) -> str:

        logger.debug(f'Assessing component "{self.component_name}" ')
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
            multinational = len(project.countries) > 1
            sources = dict()
            operators = list()
            for country in project.countries:
                if progress:
                    variable_task = progress.add_task(
                        description=f'[red]Going to process {nvars} variables', total=nvars)

                for var_name in variables:
                    var_data = variables_data[var_name]
                    if multinational:
                        operators.append(var_data['operator'])
                    #create instance
                    v = PopulationVariable(name=var_name,
                                           component=self.component_name,
                                           **var_data)

                    #assess
                    v(year=self.base_year,
                      country=country,
                      multinational=multinational,
                      **kwargs
                      )
                    if multinational:
                        sources[v.local_path] = var_name
                    if progress and variable_task:
                        progress.update(variable_task, advance=1, description=f'Assessed {var_name} in {country}')
                    #if progress and variable_task: progress.remove_task(variable_task)

            if multinational:
                vname = set(sources.values()).pop()
                operator = set(operators).pop()
                input_files = list(sources.keys())
                vrt_path = os.path.join(project.data_folder,self.component_name, vname, f'{vname}.vrt')


                vrt_options = gdal.BuildVRTOptions(
                    resolution='highest',
                    resampleAlg='nearest',
                    allowProjectionDifference=False,
                )

                with gdal.BuildVRT(destName=vrt_path, srcDSOrSrcDSTab=input_files, options=vrt_options) as vrtds:
                    vrtds.FlushCache()
                    v = PopulationVariable(
                        name=var_name,
                        component=self.component_name,
                        title=f'Multinational {vname}',
                        local_path=vrt_path,
                        operator=operator,

                    )

                    v.evaluate(multinational=multinational, year=self.base_year, **kwargs)
                    logger.info(f'{var_name} was assessed in multi-country mode {set(project.countries)}')





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
            logger.info(f'Going to compute {self.name} from {len(downloaded_files)} source files')
            computed_file = sumup(src_rasters=downloaded_files,dst_raster=local_path, overwrite=overwrite)
            assert os.path.exists(computed_file), f'The computed file: {computed_file} does not exists'
            self.local_path = computed_file
        else:
            logger.info(f'Going to compute {self.name}={self.sources}')
            src_path = self.interpolate_template(template=self.source, **kwargs)
            _, file_name = os.path.split(src_path)
            computed_file = os.path.join(self._source_folder_, file_name)
            if not os.path.exists(self._source_folder_):
                os.makedirs(self._source_folder_)
            sources = self.resolve(**kwargs)
            creation_options = 'TILED=YES COMPRESS=ZSTD BIGTIFF=IF_SAFER BLOCKXSIZE=256 BLOCKYSIZE=256 PREDICTOR=2'

            ds = Calc(calc=self.sources, outfile=computed_file,  projectionCheck=True, format='GTiff',
                      creation_options=creation_options.split(' '), quiet=False, overwrite=overwrite,  **sources)

            assert os.path.exists(computed_file), f'The computed file: {computed_file} does not exists'
            self.local_path = computed_file

    def resolve(self,  **kwargs):
        with Session() as s:
            # project = Project(path=os.getcwd())
            sources = dict()
            for var_name in self.dep_vars:
                var_dict = s.get_variable(component=self.component, variable=var_name)
                var = self.__class__(name=var_name, component=self.component, **var_dict)
                var_local_path = var(**kwargs) # assess
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
                logger.info(f'Evaluating variable {self.name} using zonal stats')
                # raster variable, run zonal stats
                src_rasters = [self.local_path]
                var_ops = [(self.name, self.operator)]
                # if project.raster_mask is not None:
                #     with gdal.OpenEx(project.geopackage_file_path) as mds, gdal.OpenEx(self.local_path) as vards:
                #
                #         l = mds.GetLayerByName('polygons')
                #         mask_srs = l.GetSpatialRef()
                #         xmin, xmax, ymin, ymax = l.GetExtent()
                #         prj_srs = vards.GetSpatialRef()
                #         if not proj_are_equal(mask_srs, prj_srs):
                #             creation_options = dict(TILED='YES', COMPRESS='ZSTD', BIGTIFF='IF_SAFER', BLOCKXSIZE=256,
                #                                     BLOCKYSIZE=256)
                #             warp_options = gdal.WarpOptions(
                #                 format='GTiff',xRes=100, yRes=100,targetAlignedPixels=True,
                #                 creationOptions=creation_options,
                #                 outputBounds=(xmin, ymin, xmax, ymax),
                #                 outputBoundsSRS=mask_srs,
                #                 dstSRS=mask_srs,
                #             )
                #             _, name = os.path.split(self.local_path)
                #             dst_path = f'/vsimem/{name}'
                #             rds = gdal.Warp(destNameOrDestDS=dst_path, srcDSOrSrcDSTab=vards,
                #                             options=warp_options)
                #
                #
                #
                #             src_rasters.append(dst_path)
                #             var_ops.append((f'{self.name}_affected', self.operator))
                # print(src_rasters)
                gdf = zst(src_rasters=src_rasters,
                                      polygon_ds=project.geopackage_file_path,
                                      polygon_layer=polygons_layer, vars_ops=var_ops
                                      )
                assert 'year' in kwargs, f'Need year kword to compute pop coeff'
                assert 'target_year' in kwargs, f'Need target_year kword to compute pop coeff'
                year = kwargs.get('year')
                target_year = kwargs.get('target_year')
                countries = set(gdf['iso3'])
                print(gdf.columns.tolist())

                for country in countries:
                    coeff = get_pop_coeff(base_year=year, target_year=target_year, country_code=country)
                    gdf.loc[gdf['iso3'] == country, self.name] *= coeff

                #gdf.rename(columns={self.name: f'{self.name}_{target_year}'}, inplace=True)

                # assert 'year' in kwargs, f'Need year kword to compute pop coeff'
                # assert 'target_year' in kwargs, f'Need target_year kword to compute pop coeff'
                # assert 'country' in kwargs, f'Need country kword to compute pop coeff'
                # year = kwargs.get('year')
                # target_year = kwargs.get('target_year')
                # country = kwargs.get('country')
                # logger.info(f'Computing pop for {target_year} with base year {year}')
                # coeff = get_pop_coeff(base_year=year, target_year=target_year, country_code=country)
                # gdf[self.name] *= coeff



            else:
                # we eval inside GeoDataFrame
                logger.debug(f'Evaluating variable {self.name} using GeoPandas eval')
                gdf = geopandas.read_file(filename=project.geopackage_file_path,layer=dst_layer)
                expr = f'{self.name}={self.sources}'
                logger.debug(expr)
                gdf.eval(expr, inplace=True)



            # merge into stats layer for component
            # Here we rely on OGR append flag. With GPKG format and because zonal_stats
            logger.debug(f'Writing columns "{", ".join(gdf.columns.tolist())}" to {project.geopackage_file_path}:{dst_layer}' )

            with io.BytesIO() as bio:

                fpath = f'/vsimem/{dst_layer}.fgb'
                write_dataframe(df=gdf,path=bio, layer=dst_layer, driver='FlatGeobuf')
                gdal.FileFromMemBuffer(fpath, bio.getbuffer())
                bio.seek(0)
                with gdal.OpenEx(fpath) as src:
                    options = gdal.VectorTranslateOptions(format='GPKG', accessMode='overwrite', layerName=dst_layer,
                                                           makeValid=True)
                    ds = gdal.VectorTranslate(destNameOrDestDS=project.geopackage_file_path,
                                              srcDS=src,options=options)

                gdal.Unlink(fpath)


