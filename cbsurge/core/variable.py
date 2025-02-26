import asyncio
import concurrent
import logging
import os.path
import random
import threading
import time
from abc import abstractmethod
from collections import deque
from typing import List
from typing import Optional, Union
import shapely
from osgeo import gdal, ogr, osr
from pyarrow import compute as pc
from pydantic import BaseModel, FilePath
from pyogrio import read_info
from rich.progress import Progress
from sympy.parsing.sympy_parser import parse_expr

from cbsurge.constants import ARROWTYPE2OGRTYPE
from cbsurge.project import Project
from cbsurge.util.downloader import downloader

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

        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

        try:
            parsed_expr = parse_expr(self.sources)

            self.dep_vars = [s.name for s in parsed_expr.free_symbols]

        except (SyntaxError, AttributeError):
            pass
        project = Project(path=os.getcwd())
        self._source_folder_ = os.path.join(project.data_folder, self.component, self.name)


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


    def download_geodata_by_admin(self, dataset_url, admin_path=None, country_col_name='iso3', admin_col_name="undp_admin_level", out_path=None,
                                batch_size=5000, NWORKERS=4):
        """
        Download a geospatial vector file with admin units as mask

        :param dataset_url: str, URL to the dataset to download. The dataset needs to be in a format that can be read by OGR and also in a cloud optimized format such as FlatGeobuf or PMTiles
        :param admin_path: str, absolute path to a geospatial vector dataset representing administrative unit names
        :param out_path: str, abs path to vector dataset to download
        :param country_col_name: str, name of the column from admin attr table that contains iso3 country code
        :param admin_col_name: str, name of the column from the admin attr table that contains amin unit name
        :param batch_size: int, defaults to 5000, the number of features to download in one batch
        :param NWORKERS, int, defaults to 4. The number of threads to use for parallel download.
        """
        assert os.path.exists(admin_path), f'Admin path {admin_path} does not exist'
        assert dataset_url, 'Dataset URL is required'
        dataset_info = read_info(dataset_url)
        assert dataset_info, f'Could not read info from {dataset_url}. Please check the URL or the dataset format'


        layer_name = dataset_info['layer_name']

        ndownloaded = 0
        failed = []
        try:
            with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
                with ogr.Open(admin_path) as adm_ds:
                    adm_lyr = adm_ds.GetLayer(0)
                    all_features = [e for e in adm_lyr]
                    stop = threading.Event()
                    jobs = deque()
                    results = deque()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=NWORKERS) as executor:
                        with Progress() as progress:
                            for feature in all_features:
                                au_geom = feature.GetGeometryRef()
                                au_poly = shapely.wkb.loads(bytes(au_geom.ExportToIsoWkb()))
                                props = feature.items()
                                au_name = props[admin_col_name]
                                job = dict(
                                    src_path=dataset_url,
                                    mask=au_poly,
                                    batch_size=batch_size,
                                    signal_event=stop,
                                    name=au_name,
                                    progress=progress

                                )
                                jobs.append(job)
                            njobs = len(jobs)
                            total_task = progress.add_task(
                                description=f'[red]Going to from {njobs} admin units', total=njobs)
                            nworkers = njobs if njobs < NWORKERS else NWORKERS
                            [executor.submit(downloader, jobs, results, stop) for i in range(nworkers)]
                            while True:
                                try:
                                    try:
                                        au_name, meta, batches = results.pop()
                                        if batches is None:
                                            raise meta
                                        logger.debug(f'{au_name} was processed')
                                        for batch in batches:
                                            if dst_ds.GetLayerCount() == 0:
                                                src_epsg = int(meta['crs'].split(':')[-1])
                                                src_srs = osr.SpatialReference()
                                                src_srs.ImportFromEPSG(src_epsg)
                                                dst_lyr = dst_ds.CreateLayer(layer_name, srs=src_srs)
                                                for name in batch.schema.names:
                                                    if 'wkb' in name or 'geometry' in name: continue
                                                    field = batch.schema.field(name)
                                                    field_type = ARROWTYPE2OGRTYPE[field.type]
                                                    dst_lyr.CreateField(ogr.FieldDefn(name, field_type))

                                            lengths = pc.binary_length(batch.column("wkb_geometry"))
                                            mask = pc.greater(lengths, 0)
                                            should_filter = pc.any(pc.invert(mask)).as_py()
                                            if should_filter:
                                                batch = batch.filter(mask)
                                            if batch.num_rows == 0:
                                                logger.debug('skipping batch')
                                                continue
                                            try:
                                                dst_lyr.WritePyArrow(batch)
                                            except Exception as e:
                                                logger.info(
                                                    f'batch with {batch.num_rows} rows from {au_name} failed with error {e} and will be ignored')

                                        dst_lyr.SyncToDisk()
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

                                except Exception as e:
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

        force_compute = kwargs.get('force_compute', False)
        if not self.dep_vars: #simple variable,
            if not force_compute:
                # logger.debug(f'Downloading {self.name} source')
                self.download(**kwargs)
            else:
                # logger.debug(f'Computing {self.name} using gdal_calc from sources')
                self.compute(**kwargs)
        else:
            if self.operator:
                if not force_compute:
                    # logger.debug(f'Downloading {self.name} from  source')
                    self.download(**kwargs)
                else:
                    # logger.debug(f'Computing {self.name}={self.sources} using GDAL')
                    self.compute(**kwargs)
            else:
                #logger.debug(f'Computing {self.name}={self.sources} using GeoPandas')
                sources = self.resolve(**kwargs)

        return self.evaluate(**kwargs)


