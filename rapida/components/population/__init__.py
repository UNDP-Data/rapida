import asyncio
import io
import os
from typing import List
import logging
import pycountry
from pyogrio import write_dataframe
from rapida.constants import POLYGONS_LAYER_NAME,GTIFF_CREATION_OPTIONS
from rapida.util import geo
from rapida.core.component import Component
from rapida.project.project import Project
from rapida.session import Session
from rapida.core.variable import Variable
from osgeo_utils.gdal_calc import Calc
from osgeo import gdal
import click
from rapida.components.population.worldpop import population_sync, process_aggregates, run_download
from rapida.stats.raster_zonal_stats import sumup, zst
import geopandas
from rapida.components.population.pop_coefficient import get_pop_coeff
from rapida.az import blobstorage
from urllib.parse import urlencode
import re

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


        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if not var_name in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        #progress = kwargs.get('progress', None)
        kwargs['computed'] = set([])
        project = Project(path=os.getcwd())
        logger.debug(f'Assessing component "{self.component_name}" in  {", ".join(project.countries)}')
        with Session() as ses:
            variables_data = ses.get_component(self.component_name)
            for var_name in variables:
                var_data = variables_data[var_name]
                #create instance
                v = PopulationVariable(
                    name=var_name,
                    component=self.component_name,
                    **var_data
                )

                #assess
                v(year=self.base_year,**kwargs)
                kwargs['computed'] |= set([var_name] + (v.dep_vars or []))




class PopulationVariable(Variable):



    def __call__(self, *args, **kwargs):
        """
                Assess a variable. Essentially this means a series of steps in a specific order:
                    - download
                    - preprocess
                    - analysis/zonal stats

                :param kwargs:
                :return:
                """

        force = kwargs.get('force', False)
        progress = kwargs.get('progress', None)

        if progress is not None:
            variable_task = progress.add_task(
                description=f'[blue]Assessing {self.component}->{self.name}', total=None)

        if not self.dep_vars:  # simple variable,
            if not force:
                # logger.debug(f'Downloading {self.name} source')
                self.download(**kwargs)
                if progress is not None and variable_task is not None:
                    progress.update(variable_task, description=f'[blue]Downloaded {self.component}->{self.name}')
            else:
                # logger.debug(f'Computing {self.name} using gdal_calc from sources')
                self.compute(**kwargs)
                if progress is not None and variable_task is not None:
                    progress.update(variable_task, description=f'[blue]Computed {self.component}->{self.name}', )

        else:
            if self.operator:
                if not force:
                    # logger.debug(f'Downloading {self.name} from  source')
                    self.download(**kwargs)
                    if progress is not None and variable_task is not None:
                        progress.update(variable_task, description=f'[blue]Downloaded {self.component}->{self.name}')
                else:
                    # logger.info(f'Computing {self.name}={self.sources} using GDAL')
                    self.compute(**kwargs)
                    if progress is not None and variable_task is not None:
                        progress.update(variable_task, description=f'[blue]Computed {self.component}->{self.name}')
            else:
                logger.debug(f'Resolving {self.name}={self.sources}')
                sources = self.resolve(**kwargs)

        self.evaluate(**kwargs)

        if progress is not None and variable_task is not None:
            progress._tasks[variable_task].total = 100
            progress.update(variable_task, description=f'[green]Assessed {self.component}->{self.name}', advance=100)


    def compute(self, **kwargs):



        var_path = os.path.join(self._source_folder_, f'{self.name}.tif')
        project = Project(os.getcwd())
        logger.debug(f'Computing {self.name} in {project.countries}')

        if not self.dep_vars:
            sources = list()
            for country in project.countries:
                # interpolate templates
                source_blobs = list()
                for source_template in self.sources:
                    source_file_path = self.interpolate_template(template=source_template, country=country, **kwargs)
                    source_blobs.append(source_file_path)
                if not os.path.exists(self._source_folder_):
                    os.makedirs(self._source_folder_)
                downloaded_files = asyncio.run(
                    blobstorage.download_blobs(src_blobs=source_blobs, dst_folder=self._source_folder_,
                                               progress=kwargs.get('progress', None))
                )
                assert len(self.sources) == len(downloaded_files), f'Not all sources were downloaded for {self.name} variable'
                src_path = self.interpolate_template(template=self.source, country=country, **kwargs)
                _, file_name = os.path.split(src_path)
                local_path = os.path.join(self._source_folder_, file_name)
                logger.debug(f'Going to compute {self.name} from {len(downloaded_files)} source files in {country}')
                computed_file = sumup(src_rasters=downloaded_files,dst_raster=local_path, overwrite=True)
                assert os.path.exists(computed_file), f'The computed file: {computed_file} does not exists'
                sources.append(computed_file)
            # build a VRT
            vrt_options = gdal.BuildVRTOptions(
                resolution='highest',
                resampleAlg='nearest',
                allowProjectionDifference=False,
            )
            vrt_path = var_path.replace('.tif', '.vrt')
            vrtds = gdal.BuildVRT(destName=vrt_path, srcDSOrSrcDSTab=sources, options=vrt_options)
            #translate to tiff
            ds = gdal.Translate(destName=var_path, srcDS=vrtds, options=['-q'])
            vrtds = None
            ds = None


        else:
            logger.debug(f'Going to compute {self.name}={self.sources}')
            if not os.path.exists(self._source_folder_):
                os.makedirs(self._source_folder_)
            sources = self.resolve( **kwargs)
            ds = Calc(calc=self.sources, outfile=var_path,  projectionCheck=True, format='GTiff',
                      creation_options=GTIFF_CREATION_OPTIONS, quiet=True, overwrite=True,  **sources)
            ds = None
            assert os.path.exists(var_path), f'The computed file: {var_path} does not exists'


        # import and compute affected version
        imported_file_path = self.import_raster(source=var_path)
        assert imported_file_path == var_path, f'var_path differs from {imported_file_path}'
        self.local_path = var_path
        self._compute_affected_()

    def resolve(self,  **kwargs):
        already_done = kwargs.get('computed', None)
        force_compute = kwargs.get('force', False)
        with Session() as s:
            sources = dict()
            for var_name in self.dep_vars:
                var_dict = s.get_variable(component=self.component, variable=var_name)
                var = self.__class__(name=var_name, component=self.component, **var_dict)
                if already_done and var_name in already_done and force_compute:
                    logger.info(f'Skipping {var_name} because it was already computed') # this will skip download by removing  force arg
                    force = kwargs.pop('force')
                var_local_path = var(**kwargs) # assess
                if already_done and var_name in already_done and force_compute:
                    kwargs['force'] = force
                sources[var_name] = var_local_path or var.local_path
            return sources


    def evaluate(self, **kwargs):

        dst_layer = f'stats.{self.component}'
        #progress = kwargs.get('progress', None)
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
                src_rasters = [self.local_path]
                var_ops = [(self.name, self.operator)]
                if project.raster_mask is not None:
                    path, file_name = os.path.split(self.local_path)
                    fname, ext = os.path.splitext(file_name)
                    affected_local_path = os.path.join(path, f'{fname}_affected{ext}')

                    src_rasters.append(affected_local_path)
                    var_ops.append((f'{self.name}_affected', self.operator))

                gdf = zst(src_rasters=src_rasters,
                                  polygon_ds=project.geopackage_file_path,
                                  polygon_layer=polygons_layer, vars_ops=var_ops
                                  )
                assert 'year' in kwargs, f'Need year kword to compute pop coeff'
                assert 'target_year' in kwargs, f'Need target_year kword to compute pop coeff'
                year = kwargs.get('year')
                target_year = kwargs.get('target_year')
                countries = set(gdf['iso3'])
                for country in countries:
                    coeff = get_pop_coeff(base_year=year, target_year=target_year, country_code=country)
                    gdf.loc[gdf['iso3'] == country, self.name] *= coeff
                    if project.raster_mask is not None:
                        gdf.loc[gdf['iso3'] == country, f'{self.name}_affected'] *= coeff
            else:
                # we eval inside GeoDataFrame
                logger.debug(f'Evaluating variable {self.name} using GeoPandas eval')
                gdf = geopandas.read_file(filename=project.geopackage_file_path,layer=dst_layer)
                expr = f'{self.name}={self.sources}'
                gdf.eval(expr, inplace=True)
                # affected
                if project.raster_mask is not None:
                    affected_sources = ''
                    for vname in self.dep_vars:
                        v = affected_sources or self.sources
                        affected_sources = v.replace(vname, f'{vname}_affected')

                    expr = f'{self.name}_affected={affected_sources}'
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

    def download(self, **kwargs):

        """Download variable"""
        logger.debug(f'Downloading {self.name} ')
        project=Project(os.getcwd())
        progress = kwargs.get('progress', None)
        self.local_path = os.path.join(self._source_folder_, f'{self.name}.tif')
        if os.path.exists(self.local_path):
            if project.raster_mask is not None:
                assert os.path.exists(self.affected_path), f'{self.affected_path} does not exist'
            return self.local_path
        sources = list()

        for country in project.countries:
            src_path = self.interpolate_template(template=self.source, country=country, **kwargs)
            _, file_name = os.path.split(src_path)
            local_path = os.path.join(self._source_folder_, file_name)
            if not os.path.exists(self._source_folder_):
                os.makedirs(self._source_folder_)
            logger.debug(f'Going to download {src_path} to {local_path}')
            downloaded_file = asyncio.run(
                blobstorage.download_blob(
                    src_path=src_path,
                    dst_path=local_path,
                    progress=progress)
            )
            sources.append(downloaded_file)

        vrt_options = gdal.BuildVRTOptions(
            resolution='highest',
            resampleAlg='nearest',
            allowProjectionDifference=False,
        )

        vrt_path = self.local_path.replace('.tif', '.vrt')
        vrtds = gdal.BuildVRT(destName=vrt_path, srcDSOrSrcDSTab=sources, options=vrt_options)
        ds = gdal.Translate(destName=self.local_path,srcDS=vrtds )
        vrtds = None
        ds = None
        imported_file_path = self.import_raster(source=self.local_path, progress=progress )
        assert imported_file_path == self.local_path, f'The local_path differs from {imported_file_path}'
        self._compute_affected_()


    def import_raster(self, source=None, **kwargs):

        project = Project(os.getcwd())
        path, file_name = os.path.split(source)
        fname, ext = os.path.splitext(file_name)
        imported_local_path = os.path.join(path, f'{fname}_imported{ext}')

        geo.import_raster(
            source=source,
            dst=imported_local_path,
            target_srs=project.target_srs,
            crop_ds=project.geopackage_file_path,
            crop_layer_name=POLYGONS_LAYER_NAME,
            **kwargs
        )

        os.remove(source)
        os.rename(imported_local_path, source)
        return source

    def _compute_affected_(self):
        project = Project(os.getcwd())
        if geo.is_raster(self.local_path) and project.raster_mask is not None :

            affected_local_path = self.affected_path
            ds = Calc(calc='local_path*mask', outfile=affected_local_path, projectionCheck=True, format='GTiff',
                      creation_options=GTIFF_CREATION_OPTIONS, quiet=False, overwrite=True,
                      NoDataValue=None,
                      local_path=self.local_path, mask=project.raster_mask)
            ds = None
            return affected_local_path

    def interpolate_template(self, template=None, **kwargs):
        """
        Interpolate values from kwargs into template
        """
        kwargs['country_lower'] = kwargs['country'].lower()
        template_vars = set(re.findall(self._extractor_, template))

        if template_vars:
            for template_var in template_vars:
                if not template_var in kwargs:
                    assert hasattr(self,template_var), f'"{template_var}"  is required to generate source files'

            return template.format(**kwargs)
        else:
            return template
    @property
    def affected_path(self):
        path, file_name = os.path.split(self.local_path)
        fname, ext = os.path.splitext(file_name)
        return os.path.join(path, f'{fname}_affected{ext}')

    def alter(self, **kwargs):
        return f'vrt://{self.local_path}?{urlencode(kwargs)}'



