import concurrent.futures

from cbsurge.util.proj_are_equal import proj_are_equal
import logging
import random
import threading
import time
from collections import deque
import numpy
import pyarrow as pa
import shapely
from osgeo import gdal, ogr, osr
from pyogrio import read_info
from pyproj import Transformer
from rich.progress import Progress
from shapely import wkb
from shapely.ops import transform
from cbsurge.constants import ARROWTYPE2OGRTYPE
from cbsurge.util.downloader import downloader, download_worker
from rich.progress import TimeElapsedColumn

logger = logging.getLogger(__name__)
gdal.UseExceptions()

OGR_TYPES_MAPPING = {
    'LineString': ogr.wkbLineString,
    'MultiLineString': ogr.wkbMultiLineString,
    'Point': ogr.wkbPoint,
    'Polygon': ogr.wkbPolygon,
    'MultiPolygon': ogr.wkbMultiPolygon,
    'GeometryCollection': ogr.wkbGeometryCollection,
    'MultiPoint': ogr.wkbMultiPoint,
}

import typing
def download_vector(
        src_dataset_url=None,src_layer_name=None,
        dst_dataset_path=None,dst_layer_name=None, dst_srs=None, overwrite_dst_layer=False,
        mask_polygons:typing.Dict[str,shapely.lib.Geometry]=None,
        batch_size=5000,NWORKERS=4,progress=None,

):
    """
    Download a remote vector dataset locally in a parallel manner using a third layer polygons
    as mask.
    :param src_dataset_url: str, URL to the dataset to download. The dataset needs to be in a format that can be read by OGR and also in a cloud optimized format such as FlatGeobuf or PMTiles
    :param: src_layer, str or int, default=0, the layer to be  read
    :param dst_dataset_path: str, local OGR dataset
    :param dst_layer_name: str, the name of src_layer in the dst_dataset that will be read
    :param mask_polygons, dict[
    :param batch_size: int, max number of features to download in one batch
    :param NWORKERS: int, default=4, number of
    :param progress:
    :return:
    """
    assert src_dataset_url not in ('', None), f'src_dataset_url={src_dataset_url} is invalid'
    assert src_layer_name not in ('', None), f'src_layer_name={src_layer_name} is invalid'

    src_dataset_info = read_info(src_dataset_url)
    assert src_dataset_info, f'Could not read info from {src_dataset_url}. Please check the URL or the dataset format'
    src_crs = src_dataset_info['crs']
    src_srs = osr.SpatialReference()
    src_srs.SetFromUserInput(src_crs)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    written_features = set()
    total_task = None
    if progress:
        cols = progress.columns
        progress.columns = [e for e in cols] + [TimeElapsedColumn()]
    try:

        with gdal.OpenEx(dst_dataset_path, gdal.OF_VECTOR | gdal.OF_UPDATE) as dst_ds:
                will_reproject =  not proj_are_equal(src_srs=src_srs, dst_srs=dst_srs)
                if will_reproject :
                    dst_crs = f"{dst_srs.GetAuthorityName(None)}:{dst_srs.GetAuthorityCode(None)}"
                    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                destination_layer = dst_ds.GetLayerByName(dst_layer_name)
                if destination_layer is not None:
                    if overwrite_dst_layer is True and dst_ds.TestCapability(ogr.ODsCDeleteLayer) is True:
                        dst_ds.DeleteLayer(destination_layer)
                else:
                    destination_layer = dst_ds.CreateLayer(
                        dst_layer_name,
                        srs=dst_srs,
                        geom_type=OGR_TYPES_MAPPING.get(src_dataset_info['geometry_type'], ogr.wkbUnknown),
                        options=['GEOMETRY_NAME=geometry']
                    )

                stop = threading.Event()
                jobs = deque()
                results = deque()
                with concurrent.futures.ThreadPoolExecutor(max_workers=NWORKERS) as executor:

                    if progress is None:
                        progress = Progress()
                    for poly_id, polygon in mask_polygons.items():
                        job = dict(
                            src_path=src_dataset_url,
                            src_layer=src_layer_name,
                            mask=polygon,
                            batch_size=batch_size,
                            signal_event=stop,
                            name=poly_id,
                            progress=progress,
                            results=results
                        )
                        jobs.append(job)
                    njobs = len(jobs)
                    total_task = progress.add_task(
                        description=f'[red]Downloading data covering {njobs} polygons', total=njobs)
                    nworkers = njobs if njobs < NWORKERS else NWORKERS
                    futures = [executor.submit(download_worker, jobs=jobs, task=total_task,stop=stop) for i in range(nworkers)]

                    nfields = destination_layer.GetLayerDefn().GetFieldCount()

                    while True:
                        try:
                            try:
                                batch = results.pop()
                                new_geometries = []
                                mask = numpy.zeros(batch.num_rows, dtype=bool)
                                for i, record in enumerate(batch.to_pylist()):
                                    geom = record.get("wkb_geometry", None)
                                    fid = record.get("OGC_FID", None)
                                    if geom is None:
                                        logger.info("Empty geometry")
                                        continue

                                    if fid in written_features:
                                        mask[i] = True
                                    else:
                                        shapely_geom = wkb.loads(geom)
                                        if will_reproject:
                                            reprojected_geom = transform(transformer.transform, shapely_geom)
                                            geom_wkb = reprojected_geom.wkb
                                        else:
                                            geom_wkb = shapely_geom.wkb
                                        written_features.add(fid)
                                        new_geometries.append(geom_wkb)
                                if mask[mask].size > 0:
                                    batch = batch.filter(~mask)

                                batch = batch.drop_columns(['wkb_geometry'])
                                batch = batch.append_column('wkb_geometry', pa.array(new_geometries))

                                if nfields == 0:
                                    logger.debug('Creating fields')
                                    for name in batch.schema.names:
                                        if 'wkb' in name or 'geometry' in name: continue
                                        field = batch.schema.field(name)
                                        field_type = ARROWTYPE2OGRTYPE[field.type]
                                        if destination_layer.GetLayerDefn().GetFieldIndex(name) == -1:
                                            destination_layer.CreateField(ogr.FieldDefn(name, field_type))

                                    nfields = destination_layer.GetLayerDefn().GetFieldCount()
                                    destination_layer.SyncToDisk()

                                if batch.num_rows == 0:
                                    logger.debug('Skipping empty batch')
                                    continue

                                batch = batch.rename_columns({"wkb_geometry": "geometry"})

                                try:
                                    destination_layer.WritePyArrow(batch)
                                    destination_layer.SyncToDisk()
                                except Exception as e:
                                    logger.info(
                                        f'writing batch with {batch.num_rows} rows from {poly_id} failed with error {e} and will be ignored')

                            except IndexError as ie:
                                done = [f.done() for f in futures]
                                if len(mask_polygons) == 0 or all(done):
                                    stop.set()
                                    break
                                s = random.random()  # this one is necessary for ^C/KeyboardInterrupt
                                time.sleep(s)
                                continue


                        except KeyboardInterrupt:
                            logger.info(f'Cancelling download. Please wait/allow for a graceful shutdown')
                            stop.set()
                            raise

    except Exception as e:
        logger.error(f'Error downloading {src_dataset_url} with error {e}')
    finally:
        if progress and total_task:
            progress.remove_task(total_task)
            progress.columns = cols





def download_geodata_by_admin(dataset_url, geopackage_path=None, batch_size=5000, NWORKERS=4, progress=None):
    """
    Download a geospatial vector file with admin units as mask

    :param geopackage_path: The path to the project geopackage file to save the downloaded data to
    :param dataset_url: str, URL to the dataset to download. The dataset needs to be in a format that can be read by OGR and also in a cloud optimized format such as FlatGeobuf or PMTiles
    :param batch_size: int, defaults to 5000, the number of features to download in one batch
    :param NWORKERS, int, defaults to 4. The number of threads to use for parallel download.
    :param progress: rich progress instance can be fetched through `kwargs.get('progress', None)`
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
        with gdal.OpenEx(geopackage_path, gdal.OF_VECTOR | gdal.OF_UPDATE) as project_dataset:
            admin_layer = project_dataset.GetLayerByName('polygons')
            admin_srs = admin_layer.GetSpatialRef()
            admin_srs_string = f"{admin_srs.GetAuthorityName(None)}:{admin_srs.GetAuthorityCode(None)}"
            transformer = Transformer.from_crs(dataset_info['crs'], admin_srs_string, always_xy=True)
            tr = osr.CoordinateTransformation(admin_srs, src_srs)
            destination_layer = project_dataset.CreateLayer(
                layer_name,
                srs=admin_srs,
                geom_type=OGR_TYPES_MAPPING.get(dataset_info['geometry_type'], ogr.wkbUnknown),
                options=['OVERWRITE=YES', 'GEOMETRY_NAME=geometry']
            )

            all_features = [e for e in admin_layer]
            stop = threading.Event()
            jobs = deque()
            results = deque()
            with concurrent.futures.ThreadPoolExecutor(max_workers=NWORKERS) as executor:
                if progress is None:
                    progress = Progress()

                for feature in all_features:
                    au_geom = feature.GetGeometryRef()
                    au_geom.Transform(tr)
                    au_poly = shapely.wkb.loads(bytes(au_geom.ExportToIsoWkb()))
                    # props = feature.items()

                    job = dict(
                        src_path=dataset_url,
                        mask=au_poly,
                        batch_size=batch_size,
                        signal_event=stop,
                        name=feature.GetFID(),
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
                                        if 'wkb' in name or 'geometry' in name: continue
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
                            if not jobs and (progress.finished or ndownloaded == len(all_features)):
                                stop.set()
                                break
                            s = random.random()  # this one is necessary for ^C/KeyboardInterrupt
                            time.sleep(s)
                            continue
                    #
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