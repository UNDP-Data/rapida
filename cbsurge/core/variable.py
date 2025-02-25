import asyncio
import concurrent
import logging
import os.path
import random
import re
import threading
import time
from abc import abstractmethod
from collections import deque
from typing import List
from typing import Optional, Union
from osgeo_utils.gdal_calc import Calc
from cbsurge import constants
from cbsurge.util import geo

import numpy
from shapely import wkb
from shapely.ops import transform
from pyproj import Transformer
import shapely
from osgeo import gdal, ogr, osr
from pyarrow import compute as pc
import pyarrow as pa
from pydantic import BaseModel, FilePath
from pyogrio import read_info
from rich.progress import Progress
from sympy.parsing.sympy_parser import parse_expr

from cbsurge.az import blobstorage
from cbsurge.constants import ARROWTYPE2OGRTYPE
from cbsurge.project import Project
from cbsurge.util.downloader import downloader
from urllib.parse import urlencode
logger = logging.getLogger(__name__)
gdal.UseExceptions()

class Variable(BaseModel):
    name: str
    title: str
    source: Optional[str] = None
    sources: Optional[Union[List[str], str]] = None
    dep_vars: Optional[List[str]] = None
    local_path: FilePath = None
    component: str = None
    operator: str = None
    _extractor_: str = r"\{([^}]+)\}"
    _default_operators_ = '+-/*%'
    _source_folder_: str = None

    def __init__(self, **kwargs):
        """
        Initialize the object with the provided arguments.
        """
        assert 'name' in kwargs, f'"name" arg is required to create a variable'
        assert 'component' in kwargs, f'"component" arg is required to create a variable'
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

        try:
            parsed_expr = parse_expr(self.sources)

            self.dep_vars = [s.name for s in parsed_expr.free_symbols]

        except (SyntaxError, AttributeError):
            pass
        project = Project(path=os.getcwd())

        self._source_folder_ = os.path.join(project.data_folder, self.component, self.name)


    # def _update_(self, **kwargs):
    #     args = self.__dict__.copy()
    #     args.update(kwargs)
    #     return args

    def __str__(self):
        """
        String representation of the class.
        """
        return f'{self.__class__.__name__} {self.model_dump_json(indent=2)}'

    @abstractmethod
    def compute(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    def alter(self, **kwargs):
        return f'vrt://{self.local_path}?{urlencode(kwargs)}'


    def download_from_azure(self, **kwargs):

        """Download variable"""
        logger.debug(f'Downloading {self.name}')
        src_path = self.interpolate_template(template=self.source, **kwargs)
        _, file_name = os.path.split(src_path)
        self.local_path = os.path.join(self._source_folder_, file_name)
        if os.path.exists(self.local_path):
            self._compute_affected_()
            return self.local_path
        if not os.path.exists(self._source_folder_):
            os.makedirs(self._source_folder_)
        logger.info(f'Going to download {self.name} from {src_path}')
        downloaded_file = asyncio.run(blobstorage.download_blob(src_path=src_path,dst_path=self.local_path))
        self.import_raster()
        assert downloaded_file == self.local_path, f'The local_path differs from {downloaded_file}'
        self._compute_affected_()


    @staticmethod
    def download_geodata_by_admin(dataset_url, geopackage_path=None, batch_size=5000, NWORKERS=4):
        """
        Download a geospatial vector file with admin units as mask

        :param geopackage_path: The path to the project geopackage file to save the downloaded data to
        :param dataset_url: str, URL to the dataset to download. The dataset needs to be in a format that can be read by OGR and also in a cloud optimized format such as FlatGeobuf or PMTiles
        :param batch_size: int, defaults to 5000, the number of features to download in one batch
        :param NWORKERS, int, defaults to 4. The number of threads to use for parallel download.
        """
        assert dataset_url, 'Dataset URL is required'
        dataset_info = read_info(dataset_url)
        assert dataset_info, f'Could not read info from {dataset_url}. Please check the URL or the dataset format'
        layer_name = dataset_info['layer_name']
        src_srs = osr.SpatialReference()
        src_srs.SetFromUserInput(dataset_info['crs'])
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ndownloaded = 0
        failed = []
        written_geometries = set()
        try:
            with gdal.OpenEx(geopackage_path, gdal.OF_VECTOR|gdal.OF_UPDATE) as project_dataset:
                admin_layer = project_dataset.GetLayerByName('polygons')
                admin_srs = admin_layer.GetSpatialRef()
                admin_srs_string = f"{admin_srs.GetAuthorityName(None)}:{admin_srs.GetAuthorityCode(None)}"
                transformer = Transformer.from_crs(dataset_info['crs'], admin_srs_string, always_xy=True)
                tr = osr.CoordinateTransformation(admin_srs, src_srs)
                destination_layer = project_dataset.CreateLayer(
                    layer_name,
                    srs=admin_srs,
                    geom_type=ogr.wkbMultiLineString,
                    options=['OVERWRITE=YES', 'GEOMETRY_NAME=geometry']
                )

                all_features = [e for e in admin_layer]
                stop = threading.Event()
                jobs = deque()
                results = deque()
                with concurrent.futures.ThreadPoolExecutor(max_workers=NWORKERS) as executor:
                    with Progress() as progress:
                        for feature in all_features:
                            au_geom = feature.GetGeometryRef()
                            au_geom.Transform(tr)
                            au_poly = shapely.wkb.loads(bytes(au_geom.ExportToIsoWkb()))
                            props = feature.items()
                            au_name = props['name'] if 'name' in props else feature.GetFID()
                            job = dict(
                                src_path=dataset_url,
                                mask=au_poly,
                                #bbox = bb,
                                batch_size=batch_size,
                                signal_event=stop,
                                name=au_name,
                                progress=progress
                            )
                            jobs.append(job)
                        njobs = len(jobs)
                        total_task = progress.add_task(
                            description=f'[red]Going to download data covering  {njobs} admin units', total=njobs)
                        nworkers = njobs if njobs < NWORKERS else NWORKERS
                        [executor.submit(downloader, jobs, results, stop) for i in range(nworkers)]
                        nfields = destination_layer.GetLayerDefn().GetFieldCount()

                        while True:
                            try:
                                try:
                                    au_name, meta, batches = results.pop()
                                    logger.info(au_name)
                                    if batches is None:
                                        logger.debug(f'{au_name} was processed')
                                        raise meta
                                    for batch in batches:
                                        new_geometries = []
                                        mask = numpy.zeros(batch.num_rows, dtype=bool)
                                        for i, record in enumerate(batch.to_pylist()):
                                            geom = record.get("wkb_geometry", None)
                                            fid = record.get("OGC_FID", None)
                                            if geom is None:
                                                print("Empty geometry")
                                                continue

                                            if fid in written_geometries:
                                                mask[i] = True
                                            else:
                                                shapely_geom = wkb.loads(geom)
                                                reprojected_geom = transform(transformer.transform, shapely_geom)
                                                geom_wkb = reprojected_geom.wkb
                                                written_geometries.add(fid)
                                                new_geometries.append(geom_wkb)
                                        if mask[mask].size > 0:
                                            batch = batch.filter(~mask)

                                        batch = batch.drop_columns(['wkb_geometry'])
                                        batch = batch.append_column('wkb_geometry', pa.array(new_geometries))

                                        if nfields == 0:
                                            logger.info('Creating fields')
                                            for name in batch.schema.names:
                                                if 'wkb' in name or 'geometry' in name:continue
                                                field = batch.schema.field(name)
                                                field_type = ARROWTYPE2OGRTYPE[field.type]
                                                if destination_layer.GetLayerDefn().GetFieldIndex(name) == -1:
                                                    destination_layer.CreateField(ogr.FieldDefn(name, field_type))

                                            nfields = destination_layer.GetLayerDefn().GetFieldCount()
                                            destination_layer.SyncToDisk()

                                        if batch.num_rows == 0:
                                            logger.info('skipping batch')
                                            continue

                                        batch = batch.rename_columns({"wkb_geometry": "geometry"})

                                        try:
                                            destination_layer.WritePyArrow(batch)
                                        except Exception as e:
                                            print(batch.column_names)
                                            logger.info(
                                                f'writing batch with {batch.num_rows} rows from {au_name} failed with error {e} and will be ignored')

                                    destination_layer.SyncToDisk()
                                    ndownloaded += 1
                                    progress.update(total_task,
                                                    description=f'[red]Downloaded geo file from {ndownloaded} out of {njobs} admin units',
                                                    advance=1)
                                except IndexError as ie:
                                    if not jobs and progress.finished:
                                        stop.set()
                                        break
                                    s = random.random()  # this one is necessary for ^C/KeyboardInterrupt
                                    time.sleep(s)
                                    continue
                            #
                            except Exception as e:
                                # import traceback
                                # traceback.print_exc(e)
                                failed.append(f'Downloading {au_name} failed: {e.__class__.__name__}("{e}")')
                                ndownloaded += 1
                                progress.update(total_task,
                                                description=f'[red]Downloaded geospatial data from {ndownloaded} out of {njobs} admin units',
                                                advance=1)
                            except KeyboardInterrupt:
                                logger.info(f'Cancelling download. Please wait/allow for a graceful shutdown')
                                stop.set()
                                break

            if failed:
                for msg in failed:
                    logger.error(msg)
        except Exception as e:
            logger.error(f'Error downloading {dataset_url} with error {e}')

    def __call__(self,  **kwargs):
            """
            Assess a variable. Essentially this means a series of steps in a specific order:
                - download
                - preprocess
                - analysis/zonal stats



            :param kwargs:
            :return:
            """
            country = kwargs.get("country")
            logger.debug(f'Assessing variable {self.name}' + (f' in {country}' if country else ''))

            force_compute = kwargs.get('force_compute', False)
            if not self.dep_vars: #simple variable,
                if not force_compute:
                    # logger.debug(f'Downloading {self.name} source')
                    self.download_from_azure(**kwargs)
                else:
                    # logger.debug(f'Computing {self.name} using gdal_calc from sources')
                    self.compute(**kwargs)
            else:
                if self.operator:
                    if not force_compute:
                        # logger.debug(f'Downloading {self.name} from  source')
                        self.download_from_azure(**kwargs)
                    else:
                        # logger.debug(f'Computing {self.name}={self.sources} using GDAL')
                        self.compute(**kwargs)
                else:
                    #logger.debug(f'Computing {self.name}={self.sources} using GeoPandas')
                    sources = self.resolve(**kwargs)
            multinational = kwargs.get('multinational', False)
            if not multinational:
                return self.evaluate(**kwargs)


    def import_raster(self):
        local_path = self.local_path
        project = Project(os.getcwd())
        path, file_name = os.path.split(local_path)
        fname, ext = os.path.splitext(file_name)
        imported_local_path = os.path.join(path, f'{fname}_imported{ext}')

        geo.import_raster(
            source=local_path,
            dst=imported_local_path,
            target_srs=project.target_srs,
            crop_ds=project.geopackage_file_path,
            crop_layer_name=constants.POLYGONS_LAYER_NAME,
        )

        os.remove(local_path)
        os.rename(imported_local_path, local_path)
        self.local_path = local_path

    def _compute_affected_(self):
        if geo.is_raster(self.local_path):
            project = Project(os.getcwd())
            path, file_name = os.path.split(self.local_path)
            fname, ext = os.path.splitext(file_name)
            affected_local_path = os.path.join(path, f'{fname}_affected{ext}')
            ds = Calc(calc='local_path*mask', outfile=affected_local_path, projectionCheck=True, format='GTiff',
                      creation_options=constants.GTIFF_CREATION_OPTIONS, quiet=False, overwrite=True,
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

