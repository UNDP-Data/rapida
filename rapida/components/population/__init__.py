import asyncio
import io
import os
from typing import List
import logging
import rasterio
from pyogrio import write_dataframe
from pyproj import Transformer
from rasterio.windows import from_bounds

from rapida.constants import POLYGONS_LAYER_NAME,GTIFF_CREATION_OPTIONS
from rapida.util import geo
from rapida.core.component import Component
from rapida.project.project import Project
from rapida.session import Session
from rapida.core.variable import Variable
from osgeo_utils.gdal_calc import Calc
from osgeo import gdal
from rapida.components.population.worldpop import population_sync, process_aggregates, run_download
from rapida.stats.raster_zonal_stats import sumup, zst
import geopandas
from rapida.components.population.pop_coefficient import get_pop_coeff
from urllib.parse import urlencode
import re
from rapida.util.download_remote_file import download_remote_files


wpop_countries = [
    "ABW", "AFG", "AGO", "AIA", "ALA", "ALB", "AND", "ARE", "ARG", "ARM", "ASM", "ATA", "ATF", "ATG",
    "AUS", "AUT", "AZE", "BDI", "BEL", "BEN", "BES", "BFA", "BGD", "BGR", "BHR", "BHS", "BIH", "BLM",
    "BLR", "BLZ", "BMU", "BOL", "BRA", "BRB", "BRN", "BTN", "BVT", "BWA", "CAF", "CAN", "CHE", "CHL",
    "CHN", "CIV", "CMR", "COD", "COG", "COK", "COL", "COM", "CPV", "CRI", "CUB", "CUW", "CYM", "CYP",
    "CZE", "DEU", "DJI", "DMA", "DNK", "DOM", "DZA", "ECU", "EGY", "ERI", "ESH", "ESP", "EST", "ETH",
    "FIN", "FJI", "FLK", "FRA", "FRO", "FSM", "GAB", "GBR", "GEO", "GGY", "GHA", "GIB", "GIN", "GLP",
    "GMB", "GNB", "GNQ", "GRC", "GRD", "GRL", "GTM", "GUF", "GUM", "GUY", "HKG", "HMD", "HND", "HRV",
    "HTI", "HUN", "IDN", "IMN", "IND", "IOT", "IRL", "IRN", "IRQ", "ISL", "ISR", "ITA", "JAM", "JEY",
    "JOR", "JPN", "KAZ", "KEN", "KGZ", "KHM", "KIR", "KNA", "KOR", "KOS", "KWT", "LAO", "LBN", "LBR",
    "LBY", "LCA", "LIE", "LKA", "LSO", "LTU", "LUX", "LVA", "MAC", "MAF", "MAR", "MCO", "MDA", "MDG",
    "MDV", "MEX", "MHL", "MKD", "MLI", "MLT", "MMR", "MNE", "MNG", "MNP", "MOZ", "MRT", "MSR", "MTQ",
    "MUS", "MWI", "MYS", "MYT", "NAM", "NCL", "NER", "NFK", "NGA", "NIC", "NIU", "NLD", "NOR", "NPL",
    "NRU", "NZL", "OMN", "PAK", "PAN", "PCN", "PER", "PHL", "PLW", "PNG", "POL", "PRI", "PRK", "PRT",
    "PRY", "PSE", "PYF", "QAT", "REU", "ROU", "RUS", "RWA", "SAU", "SDN", "SEN", "SGP", "SGS", "SHN",
    "SJM", "SLB", "SLE", "SLV", "SMR", "SOM", "SPM", "SPR", "SRB", "SSD", "STP", "SUR", "SVK", "SVN",
    "SWE", "SWZ", "SXM", "SYC", "SYR", "TCA", "TCD", "TGO", "THA", "TJK", "TKL", "TKM", "TLS", "TON",
    "TTO", "TUN", "TUR", "TUV", "TWN", "TZA", "UGA", "UKR", "UMI", "URY", "USA", "UZB", "VAT", "VCT",
    "VEN", "VGB", "VIR", "VNM", "VUT", "WLF", "WSM", "YEM", "ZAF", "ZMB", "ZWE"
]


logger = logging.getLogger('rapida')

gdal.UseExceptions()



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
                if country not in wpop_countries:
                    logger.info(f"The population dataset for country {country} is missing. The download will therefore skip")
                    continue
                # interpolate templates
                source_blobs = list()
                for source_template in self.sources:
                    source_file_path = self.interpolate_template(template=source_template, country=country, **kwargs)
                    if source_file_path.startswith('az:'):
                        # if start with az:, interpolate it to HTTP URL
                        proto, account_name, src_blob_path = source_file_path.split(':')
                        base_url = f"https://{account_name}.blob.core.windows.net"
                        source_file_path = f"{base_url}/{src_blob_path}"

                    source_blobs.append(source_file_path)
                if not os.path.exists(self._source_folder_):
                    os.makedirs(self._source_folder_)

                downloaded_files = asyncio.run(download_remote_files(source_blobs, self._source_folder_, progress=kwargs.get('progress', None)))

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
        self._compute_affected_(**kwargs)

    def resolve(self,  **kwargs):
        already_done = kwargs.get('computed', None)
        force_compute = kwargs.get('force', False)
        with Session() as s:
            sources = dict()
            for var_name in self.dep_vars:
                var_dict = s.get_variable(component=self.component, variable=var_name)
                var = self.__class__(name=var_name, component=self.component, **var_dict)
                if already_done and var_name in already_done and force_compute:
                    logger.debug(f'Skipping {var_name} because it was already computed') # this will skip download by removing  force arg
                    force = kwargs.pop('force')
                var_local_path = var(**kwargs) # assess
                if already_done and var_name in already_done and force_compute:
                    kwargs['force'] = force
                sources[var_name] = var_local_path or var.local_path
            return sources


    def evaluate(self, **kwargs):

        dst_layer = f'stats.{self.component}'
        progress = kwargs.get('progress', None)
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
                year = kwargs.get('year')
                target_year = kwargs.get('target_year')
                # raster variable, run zonal stats
                src_rasters = [self.local_path]
                var_ops = [(f"{target_year}_{self.name}", self.operator)]
                if project.raster_mask is not None:
                    path, file_name = os.path.split(self.local_path)
                    fname, ext = os.path.splitext(file_name)
                    affected_local_path = os.path.join(path, f'{fname}_affected{ext}')

                    src_rasters.append(affected_local_path)
                    var_ops.append((f'{target_year}_{self.name}_affected', self.operator))
                print(var_ops)
                gdf = zst(src_rasters=src_rasters,
                                  polygon_ds=project.geopackage_file_path,
                                  polygon_layer=polygons_layer, vars_ops=var_ops, progress=progress
                                  )
                assert 'year' in kwargs, f'Need year kword to compute pop coeff'
                assert 'target_year' in kwargs, f'Need target_year kword to compute pop coeff'

                countries = set(gdf['iso3'])
                for country in countries:
                    coeff = get_pop_coeff(base_year=year, target_year=target_year, country_code=country)
                    gdf.loc[gdf['iso3'] == country, f'{target_year}_{self.name}'] *= coeff
                    if project.raster_mask is not None:
                        gdf.loc[gdf['iso3'] == country, f'{target_year}_{self.name}_affected'] *= coeff
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
        logger.debug(f'Downloading {self.name}')
        project = Project(os.getcwd())
        progress = kwargs.get('progress', None)
        self.local_path = os.path.join(self._source_folder_, f'{self.name}.tif')

        if os.path.exists(self.local_path):
            if project.raster_mask is not None:
                assert os.path.exists(self.affected_path), f'{self.affected_path} does not exist'
            return self.local_path

        sources = []

        for country in project.countries:
            if country not in wpop_countries:
                logger.info(f"The population dataset for country {country} is missing. The download will therefore skip")
                continue
            src_path = self.interpolate_template(template=self.source, country=country, **kwargs)
            _, file_name = os.path.split(src_path)
            local_path = os.path.join(self._source_folder_, file_name)
            os.makedirs(self._source_folder_, exist_ok=True)
            logger.debug(f'Going to download {src_path} to {local_path}')

            if src_path.startswith('az:'):
                # if start with az:, interpolate it to HTTP URL
                proto, account_name, src_blob_path = src_path.split(':')
                base_url = f"https://{account_name}.blob.core.windows.net"
                src_path = f"{base_url}/{src_blob_path}"

            sources.append(src_path)
        downloaded_files = asyncio.run(download_remote_files(sources, self._source_folder_, progress=progress))

        vrt_path = self.local_path.replace('.tif', '.vrt')
        vrt_options = gdal.BuildVRTOptions(
            resolution='highest',
            resampleAlg='nearest',
            allowProjectionDifference=False,
        )
        vrtds = gdal.BuildVRT(destName=vrt_path, srcDSOrSrcDSTab=downloaded_files, options=vrt_options)
        vrtds.FlushCache()
        polygons_ds = gdal.OpenEx(project.geopackage_file_path, gdal.OF_VECTOR)
        polygons_layer = polygons_ds.GetLayerByName(project.polygons_layer_name)
        bounds = polygons_layer.GetExtent()  # (minx, maxx, miny, maxy)
        minx, maxx, miny, maxy = bounds

        with rasterio.open(vrt_path) as ras:
            raster_crs = ras.crs
            raster_transform = ras.transform
            transformer = Transformer.from_crs(project.target_srs.ExportToWkt(), raster_crs.to_wkt(), always_xy=True)
            left, bottom, right, top = transformer.transform_bounds(left=minx, bottom=miny, right=maxx, top=maxy)

            window = from_bounds(left, bottom, right, top, transform=raster_transform)
            xoff, yoff = int(window.col_off), int(window.row_off)
            xsize, ysize = int(window.width), int(window.height)

        ds = gdal.Translate(
            destName=self.local_path,
            srcDS=vrtds,
            srcWin=[xoff, yoff, xsize, ysize]
        )
        vrtds = None
        polygons_ds = None
        ds = None
        imported_file_path = self.import_raster(source=self.local_path, progress=progress)
        assert imported_file_path == self.local_path, f'The local_path differs from {imported_file_path}'
        self._compute_affected_(progress=progress)


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

    def _compute_affected_(self, progress=None, **kwargs):
        project = Project(os.getcwd())
        if geo.is_raster(self.local_path) and project.raster_mask is not None:
            if progress is not None:
                task = progress.add_task(description="Computing affected version using GDAL calc")

            def progress_callback(complete, message, data, progress=progress, task=task):
                if progress is not None and task is not None:
                    progress.update(task, completed=int(complete * 100))

            affected_local_path = self.affected_path

            ds = Calc(calc='local_path*mask', outfile=affected_local_path, projectionCheck=True, format='GTiff',
                      creation_options=GTIFF_CREATION_OPTIONS, quiet=False, overwrite=True,
                      NoDataValue=None,
                      local_path=self.local_path, mask=project.raster_mask, progress_callback=progress_callback)
            ds = None
            progress.remove_task(task)
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



