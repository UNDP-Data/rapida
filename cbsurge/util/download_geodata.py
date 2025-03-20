import concurrent.futures
import logging
import random
import time
from collections import deque
import concurrent.futures
import threading
import numpy
import pyarrow as pa
import shapely
from osgeo import gdal, ogr, osr
from pyogrio import read_info
from pyproj import Transformer
from rich.progress import Progress
from shapely import wkb
import rasterio
from rasterio.windows import from_bounds, Window
from cbsurge.util.proj_are_equal import proj_are_equal
from cbsurge import constants
from shapely.ops import transform
from cbsurge.constants import ARROWTYPE2OGRTYPE
from cbsurge.util.downloader import downloader
from cbsurge.util.worker import worker
from cbsurge.util.read_bbox import stream, read_rasterio_window
from cbsurge.util.generator_length import generator_length
from rich.progress import TimeElapsedColumn
import typing
from cbsurge.util.gen_blocks_bbox import gen_blocks_bbox
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)
gdal.UseExceptions()


OGR_GEOM_TYPES_MAPPING = {
    'LineString': ogr.wkbLineString,
    'MultiLineString': ogr.wkbMultiLineString,
    'Point': ogr.wkbPoint,
    'Polygon': ogr.wkbPolygon,
    'MultiPolygon': ogr.wkbMultiPolygon,
    'GeometryCollection': ogr.wkbGeometryCollection,
    'MultiPoint': ogr.wkbMultiPoint,
}

PYTHON2OGR = {
    str: ogr.OFTString,
    int: ogr.OFTInteger64,  # Python ints are typically 64-bit
    float: ogr.OFTReal,
    bool: ogr.OFTInteger,  # OGR has no boolean type, so 0/1 integer
    bytes: ogr.OFTBinary,
}


def download_vector(
        src_dataset_url=None,src_layer_name=None,
        dst_dataset_path=None,dst_layer_name=None, dst_srs=None, dst_layer_mode=None,
        mask_polygons:typing.Dict[str,shapely.lib.Geometry]=None, add_polyid=False,
        batch_size=5000,NWORKERS=4,progress=None,

):
    """
    Download a remote vector dataset locally in a parallel manner using polygons as  a spatial mask. Uses PyArrow streaming.

    :param src_dataset_url: str, URL to the dataset to download. The dataset needs to be in a format that can be read by OGR and also in a cloud optimized format such as FlatGeobuf or PMTiles. The URL should start with `/vsicurl/` prefix, if not, the funciton will add the prefix to proceed.
    :param: src_layer_name, str or int, default=0, the layer to be  read
    :param dst_dataset_path: str, local OGR dataset
    :param dst_layer_name: str, the name of src_layer in the dst_dataset that will be read
    :param dst_srs instance of osr.SpatialReference
    :param dst_layer_mode, str, default='w', if w layer is created and if it exists deleted, else features are appended to the layer
    :param mask_polygons,  a dict used to limit the download spatially.
    :param add_polyid: bool, if True the poygon id supplied as key in the mask_polygons arg is set for every downloaded feature
    :param batch_size: int, max number of features to download in one batch
    :param NWORKERS: int, default=4, number of
    :param progress:
    :return:
    """
    assert src_dataset_url not in ('', None), f'src_dataset_url={src_dataset_url} is invalid'
    assert src_layer_name not in ('', None), f'src_layer_name={src_layer_name} is invalid'


    def _create_layer_(_src_layer_=None, _dst_ds_=None, _dst_layer_name_=None, _dst_srs_=None):
        _src_layer_defn_ = _src_layer_.GetLayerDefn()
        _destination_layer_ = _dst_ds_.CreateLayer(
            _dst_layer_name_,
            srs=_dst_srs_,
            geom_type=_src_layer_defn_.GetGeomType(),
            options=['OVERWRITE=YES', 'GEOMETRY_NAME=geometry', 'FID=FID']
        )
        # Copy fields
        for i in range(_src_layer_defn_.GetFieldCount()):
            field_defn = _src_layer_defn_.GetFieldDefn(i)
            _destination_layer_.CreateField(field_defn)

        return _destination_layer_


    written_features = set()
    total_task = None
    if progress:
        cols = progress.columns
        progress.columns = [e for e in cols] + [TimeElapsedColumn()]
    try:
        stop = None
        with gdal.OpenEx(dst_dataset_path, gdal.OF_VECTOR | gdal.OF_UPDATE) as dst_ds:
            if not src_dataset_url.startswith('/vsicurl/'):
                src_dataset_url = f"/vsicurl/{src_dataset_url}"
            with gdal.OpenEx(src_dataset_url, gdal.OF_VECTOR | gdal.OF_READONLY) as src_ds:
                if isinstance(src_layer_name, int) or src_layer_name is None:
                    layer_index = src_layer_name if src_layer_name is not None else 0
                    src_layer = src_ds.GetLayer(layer_index)
                else:
                    src_layer = src_ds.GetLayerByName(src_layer_name)
                src_srs = src_layer.GetSpatialRef()
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                will_reproject =  not proj_are_equal(src_srs=src_srs, dst_srs=dst_srs)
                if will_reproject :
                    src_crs = f"{src_srs.GetAuthorityName(None)}:{src_srs.GetAuthorityCode(None)}"
                    dst_crs = f"{dst_srs.GetAuthorityName(None)}:{dst_srs.GetAuthorityCode(None)}"
                    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                if dst_layer_mode == 'w':
                    destination_layer = _create_layer_(
                        _src_layer_=src_layer,
                        _dst_ds_=dst_ds,
                        _dst_layer_name_=dst_layer_name,
                        _dst_srs_=dst_srs
                    )
                elif dst_layer_mode == 'a':
                    destination_layer = dst_ds.GetLayerByName(dst_layer_name)
                    if destination_layer is None:
                        destination_layer = _create_layer_(
                            _src_layer_=src_layer,
                            _dst_ds_=dst_ds,
                            _dst_layer_name_=dst_layer_name,
                            _dst_srs_=dst_srs
                        )
                field_names = [destination_layer.GetLayerDefn().GetFieldDefn(i).GetName() for i in
                               range(destination_layer.GetLayerDefn().GetFieldCount())]
                if add_polyid and not 'polyid' in field_names:
                    first_poly_id = list(mask_polygons.keys())[0]
                    first_poly_type = type(first_poly_id)
                    assert all([isinstance(e, first_poly_type) for e in mask_polygons]), f'mask polygon keys needs to be {first_poly_type}'
                    poly_field = ogr.FieldDefn('polyid', PYTHON2OGR[first_poly_type])
                    destination_layer.CreateField(poly_field)
                if not 'SRC_FID' in field_names:
                    srcid_field = ogr.FieldDefn('SRC_FID', ogr.OFTInteger64)
                    destination_layer.CreateField(srcid_field)


                stop = threading.Event()
                jobs = deque()
                results = deque()
                with concurrent.futures.ThreadPoolExecutor(max_workers=NWORKERS) as executor:

                    if progress is None:
                        progress = Progress()
                    for poly_id, polygon in mask_polygons.items():
                        job = dict(
                            src_path=src_dataset_url,
                            src_layer=src_layer.GetName(),
                            mask=polygon,
                            batch_size=batch_size,
                            signal_event=stop,
                            polygon_id=poly_id,
                            progress=progress,
                            results=results,
                            add_polyid=add_polyid
                        )
                        jobs.append(job)

                    ndownloaded = 0
                    njobs = len(jobs)
                    total_task = progress.add_task(
                        description=f'[red]Downloading data covering {njobs} polygons', total=njobs)
                    nworkers = njobs if njobs < NWORKERS else NWORKERS
                    futures = [executor.submit(worker, job=stream, jobs=jobs, task=total_task,stop=stop, id_prop_name='polygon_id') for i in range(nworkers)]
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
                                if batch.num_rows == 0:
                                    logger.debug('Skipping empty batch')
                                    continue

                                batch = batch.rename_columns({"wkb_geometry": "geometry", 'OGC_FID':'SRC_FID'})

                                try:
                                    destination_layer.WritePyArrow(batch)
                                    destination_layer.SyncToDisk()
                                except Exception as e:
                                    logger.info(
                                        f'writing batch with {batch.num_rows} rows from {poly_id} failed with error {e} and will be ignored')
                                # keep the number of worker processed to prevent going into infinite loop
                                ndownloaded += 1

                            except IndexError as ie:
                                done = [f.done() for f in futures]
                                if len(mask_polygons) == 0 or all(done) or ndownloaded == len(jobs):
                                    break
                                s = random.random()  # this one is necessary for ^C/KeyboardInterrupt
                                time.sleep(s)
                                continue


                        except KeyboardInterrupt:
                            logger.info(f'Cancelling download. Please wait/allow for a graceful shutdown and cleanup')
                            stop.set()
                            raise

    except Exception as e:
        logger.error(f'Error downloading {src_dataset_url} with error {e}')
        raise
    finally:
        if progress is not None and total_task:
            progress.remove_task(total_task)
            progress.columns = cols
        if stop is not None and stop.is_set(): # the download was cancelled, the layer is removed. Should this be?
            with gdal.OpenEx(dst_dataset_path, gdal.OF_VECTOR | gdal.OF_UPDATE) as target_ds:
                for i in  range(target_ds.GetLayerCount()):
                    l = target_ds.GetLayer(i)
                    if l.GetName() == dst_layer_name:
                        logger.info(f'Layer {dst_layer_name} will be deleted as a result of cancelling')

                        target_ds.DeleteLayer(i)




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
                geom_type=OGR_GEOM_TYPES_MAPPING.get(dataset_info['geometry_type'], ogr.wkbUnknown),
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


def download_raster( src_dataset_path=None, src_band=1,
                    dst_dataset_path=None, dst_band=1,
                    polygons_dataset_path=None, polygons_layer_name=None,
                    horizontal_chunk_size=256, vertical_chunk_size=256, workers=4, progress=None
                   ):

    with gdal.OpenEx(polygons_dataset_path, gdal.OF_VECTOR|gdal.OF_READONLY) as poly_ds:
        lyr = poly_ds.GetLayerByName(polygons_layer_name)
        minx, maxx, miny, maxy = lyr.GetExtent(force=True)
        dst_srs = lyr.GetSpatialRef()
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst_crs = f"{dst_srs.GetAuthorityName(None)}:{dst_srs.GetAuthorityCode(None)}"
        if not src_dataset_path.startswith('/vsicurl'):
            src_dataset_path = f'/vsicurl/{src_dataset_path}'
        #with gdal.OpenEx(src_dataset_path, gdal.OF_READONLY | gdal.OF_RASTER) as src_ds:
        with rasterio.open(src_dataset_path, mode='r') as src_ds:
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(src_ds.crs.to_wkt())
            #src_srs = src_ds.GetSpatialRef()
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            should_reproject = not proj_are_equal(src_srs=src_srs, dst_srs=dst_srs)
            if should_reproject:
                bounds = transform_bounds(src_crs=CRS.from_wkt(dst_srs.ExportToWkt()),
                                          dst_crs=src_ds.crs,
                                          left=minx,bottom=miny,right=maxx, top=maxy, )

            else:
                bounds = minx, miny, maxx, maxy

            win = rasterio.windows.from_bounds(*bounds,transform=src_ds.transform)
            #print(src_ds.profile)
            row_slice, col_slice = win.toslices()
            src_ds.RasterXSize = src_ds.width
            src_ds.RasterYSize = src_ds.height
            blocks = gen_blocks_bbox(ds=src_ds,blockxsize=horizontal_chunk_size, blockysize=vertical_chunk_size,
                                     xminc=col_slice.start,yminr=row_slice.start,
                                     xmaxc=col_slice.stop, ymaxr=row_slice.stop)

            nblocks, blocks = generator_length(blocks)

            block_dict = {}
            stop_event = threading.Event()
            jobs = deque()
            results = deque()
            workers = nblocks if workers > nblocks else workers
            profile = src_ds.meta.copy()
            profile['width'] = abs(col_slice.start-col_slice.stop)
            profile['height'] = abs(row_slice.start-row_slice.stop)
            profile.update(constants.GTIFF_CREATION_OPTIONS)
            profile['transform'] = rasterio.transform.from_bounds(*bounds,profile['width'], profile['height'])

            with rasterio.open(dst_dataset_path, mode='w+', **profile) as dst_ds:

                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:

                    if progress:
                        total_task = progress.add_task(
                            description=f'[red]Going to download data in {nblocks} blocks/chunks',
                            total=nblocks)
                    for block_id, block in enumerate(blocks):
                        win_id = f'window_{block_id}'
                        job = dict(
                            src_ds_path=src_dataset_path,
                            src_band=src_band,
                            window=Window(*block),
                            window_id=win_id,
                            results=results,
                            progress=progress

                        )
                        jobs.append(job)
                        block_dict[win_id] = block

                    futures = [
                        executor.submit(worker, job=read_rasterio_window, jobs=jobs, stop=stop_event, task=total_task,
                                        id_prop_name='window_id') for i in range(workers)]

                    while True:
                        try:
                            try:
                                win_id, data = results.pop()
                                block = block_dict[win_id]
                                col_start, row_start, col_size, row_size = block
                                write_win = Window(
                                    col_off=col_start-col_slice.start,
                                    row_off = row_start-row_slice.start,
                                    width=col_size,
                                    height=row_size
                                )

                                dst_ds.write(data,dst_band, window=write_win)
                                try:
                                    pass # write here to dst raster
                                except Exception as e:
                                    logger.error(
                                        f'Failed to write  block id {block_id} because {e}. Skipping')

                            except IndexError as ie:
                                done = [f.done() for f in futures]
                                if nblocks == 0 or all(done):
                                    stop_event.set()
                                    break
                                s = random.random()  # this one is necessary for ^C/KeyboardInterrupt
                                time.sleep(s)
                                continue
                        except KeyboardInterrupt:
                            logger.info(f'Cancelling. Please wait/allow for a graceful shutdown')
                            stop_event.set()
                            raise



