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
import tempfile
from osgeo import gdal, ogr, osr
from pyogrio import read_info
from pyproj import Transformer
from rich.progress import Progress
from shapely import wkb
from shapely.ops import transform, unary_union
from shapely.geometry import box
import os
import rasterio
from rasterio.windows import from_bounds, Window
import geopandas as gpd
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
from affine import Affine
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


def download_raster(
        dataset_url: str,
        geopackage_path: str,
        output_filename: str,
        mask_layer_name: str = None,
        progress: Progress=None,
        chunk_size: tuple[int, int]=(4096, 4096),
        num_workers=4,
        gdal_cache_max=1024*1024*1024
):
    """
    download raster data clipped by the project area
    This function performs two steps:
      1. During concurrent processing, it writes chunks to a single intermediate file
         in the source dataset's coordinate system.
      2. After concurrent processing is complete, creates a VRT including all intermediate files to merge them to a single file

    :param dataset_url: raster dataset url
    :param geopackage_path: project geopackage path. It should contain `polygons` layer unless set layer name as mask_layer_name
    :param output_filename: output filename to save the downloaded raster file
    :param mask_layer_name: mask layer name. If not specified, it clips the raster by `polygons` layer.
    :param progress: rich progress instance
    :param chunk_size: A tuple (chunk_width, chunk_height) in pixels.
    :param num_workers: Number of concurrent threads.
    :param gdal_cache_max: Max cache size for GDAL
    :return: output file path is returned
    """
    download_task = None
    if progress:
        download_task = progress.add_task(
            description='Downloading data covering the project area', total=None)

    output_dir = os.path.dirname(geopackage_path)
    mask_layer = constants.POLYGONS_LAYER_NAME
    if mask_layer_name:
        mask_layer = mask_layer_name
    gdf = gpd.read_file(geopackage_path, layer=mask_layer)
    polygons_crs = gdf.crs

    with rasterio.open(dataset_url) as src:
        src_nodata = src.nodata
        # Reproject the project polygons to match the source raster CRS.
        gdf_reproj = gdf.to_crs(src.crs)
        geoms = gdf_reproj.geometry.values.tolist()

        if progress and download_task is not None:
            progress.update(download_task, description=f'Identified {len(geoms)} polygon(s) for masking.')

        # Create a single unioned geometry from all polygons for efficiency.
        union_geom = unary_union(geoms)
        minx, miny, maxx, maxy = union_geom.bounds

        # Compute the overall window in pixel coordinates covering the unioned bounds.
        overall_window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)

        # Set chunk size in pixels.
        chunk_width, chunk_height = chunk_size

        # Get integer bounds of the overall window.
        col_off = int(overall_window.col_off)
        row_off = int(overall_window.row_off)
        win_width = int(overall_window.width)
        win_height = int(overall_window.height)

        # Create a list of windows (chunks) relative to the overall_window.
        rows = range(row_off, row_off + win_height, chunk_height)
        cols = range(col_off, col_off + win_width, chunk_width)
        windows = [
            Window(
                col - col_off,
                row - row_off,
                min(chunk_width, win_width - (col - col_off)),
                min(chunk_height, win_height - (row - row_off))
            )
            for row in rows for col in cols
        ]
        total_chunks = len(windows)
        if progress and download_task is not None:
            progress.update(download_task, total=total_chunks, description=f'Downloading {total_chunks} chunks')

        # Prepare metadata for the intermediate output file (in source CRS).
        out_transform = rasterio.windows.transform(overall_window, src.transform)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "COG",
            "crs": src.crs,
            "transform": out_transform,
            "width": int(overall_window.width),
            "height": int(overall_window.height),
            "compress": "zstd",
            "predictor": 2
        })

    if progress and download_task is not None:
        progress.update(download_task,
                        description=f'Processing {total_chunks} chunks with {num_workers} threads.',
                        total=None)

    # Create a temporary directory for all temporary files.
    with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
        # List to store temporary file paths and their corresponding mosaic windows.
        temp_files_data = []
        temp_files_lock = threading.Lock()

        chunk_progress = {}
        cancel_event = threading.Event()

        def process(mosaic_window: Window, task_id):
            """
            Process one chunk:
              1. Convert mosaic_window (relative to overall_window) to a source absolute window.
              2. Check intersection with the unioned polygon.
              3. Read source data, apply mask, and write directly into the intermediate file.
            """
            if cancel_event.is_set():
                return
            chunk_progress[task_id] = {"progress": 0, "total": 1}

            # Convert mosaic_window (relative to overall_window) to source coordinates.
            source_window = Window(
                mosaic_window.col_off + overall_window.col_off,
                mosaic_window.row_off + overall_window.row_off,
                mosaic_window.width,
                mosaic_window.height
            )
            with rasterio.open(dataset_url) as src:
                data = src.read(window=source_window)

                # Get the bounds of the source window.
                s_bounds = rasterio.windows.bounds(source_window, transform=src.transform)
                # Check if the source window intersects the unioned polygon.
                if not box(*s_bounds).intersects(union_geom):
                    # Create an array filled with nan for all bands.
                    data = numpy.full(
                        (src.count, int(mosaic_window.height), int(mosaic_window.width)),
                        numpy.nan,
                        dtype=src.dtypes[0]
                    )
                else:
                    src_window_transform = src.window_transform(source_window)
                    # Create a mask array using the source window dimensions.
                    mask_arr = rasterio.features.geometry_mask(
                        geoms,
                        out_shape=(int(source_window.height), int(source_window.width)),
                        transform=src_window_transform,
                        invert=True
                    )
                    data = numpy.where(mask_arr, data, numpy.nan)

            # Determine the temporary file path for the current chunk.
            temp_filename = os.path.join(tmpdir, f"{output_filename}_{task_id}.tmp")
            # Calculate the transform for the current chunk relative to the overall window.
            block_transform = rasterio.windows.transform(mosaic_window, out_transform)
            # Prepare metadata for the temporary file.
            block_meta = out_meta.copy()
            block_meta.update({
                "width": int(mosaic_window.width),
                "height": int(mosaic_window.height),
                "transform": block_transform
            })
            # Write the processed chunk data to the temporary file.
            with rasterio.open(temp_filename, "w", **block_meta) as temp_dst:
                temp_dst.write(data)
            # Store the temporary file path and its window information.
            with temp_files_lock:
                temp_files_data.append(temp_filename)
            chunk_progress[task_id] = {"progress": 1, "total": 1}


        futures = []
        chunk_task_ids = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for idx, mosaic_window in enumerate(windows):
                t_id = progress.add_task(f"Processing chunk {idx}: {mosaic_window.col_off}, {mosaic_window.row_off}", visible=False, total=None)
                chunk_task_ids.append(t_id)
                futures.append(executor.submit(process, mosaic_window, t_id))
            overall_chunks_task = progress.add_task("[green]All chunks progress:", total=len(futures))
            try:
                while any(not f.done() for f in futures):
                    n_finished = sum(f.done() for f in futures)
                    progress.update(overall_chunks_task, completed=n_finished, total=len(futures))
                    for t_id in chunk_task_ids:
                        if t_id in chunk_progress:
                            latest = chunk_progress[t_id]["progress"]
                            total = chunk_progress[t_id]["total"]
                            progress.update(t_id, visible=latest < total)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                if progress and download_task is not None:
                    progress.update(download_task,
                                    description=f'Cancelling downloading processes...',
                                    advance=total_chunks)
                cancel_event.set()

            for future in futures:
                if not future.done():
                    future.cancel()

            progress.update(overall_chunks_task, completed=len(futures), total=len(futures))
            for t_id in chunk_task_ids:
                progress.remove_task(t_id)
            progress.remove_task(overall_chunks_task)

        if cancel_event.is_set():
            # Remove all temporary files if processing was cancelled.
            with temp_files_lock:
                for temp_file, _ in temp_files_data:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            if progress and download_task is not None:
                progress.update(download_task,
                                description=f'Download cancelled by user.',
                                advance=total_chunks)
            raise KeyboardInterrupt("Download cancelled by user.")

        if progress and download_task is not None:
            progress.update(download_task,
                            description=f'[red]All chunks were downloaded.',
                            advance=total_chunks)
            progress.remove_task(download_task)

        merge_task = None
        if progress:
            merge_task = progress.add_task("[green]Merging files...", total=100)

        # Define a callback function for GDAL that updates the Rich progress task.
        def progress_callback(complete, message, user_data):
            if progress and merge_task is not None:
                progress.update(merge_task, completed=int(complete * 100))
            return 1

        if progress and merge_task is not None:
            progress.update(merge_task, description="[green]Creating VRT...")
        vrt_filename = os.path.join(tmpdir, f"{output_filename}.vrt")
        with gdal.BuildVRT(vrt_filename, temp_files_data,
                           resampleAlg='nearest', srcNodata=src_nodata) as vrt:
            if progress and merge_task is not None:
                progress.update(merge_task, description=f"[green]Writing COG to {output_filename}...")
            warp_options = gdal.WarpOptions(
                dstSRS=polygons_crs.to_wkt(),
                format="COG",
                creationOptions=["COMPRESS=ZSTD", "PREDICTOR=2","BIGTIFF=IF_SAFER", "BLOCKSIZE=256"],
                resampleAlg='nearest',
                srcNodata=src_nodata,
                callback=progress_callback
            )
            gdal.SetCacheMax(gdal_cache_max)
            final_output_path = os.path.join(output_dir, output_filename)
            gdal.Warp(final_output_path, vrt, options=warp_options)

        if progress and merge_task:
            progress.update(merge_task,
                            description=f'[red]Saved to {final_output_path}.',
                            total=None)
            progress.remove_task(merge_task)

    return final_output_path

def download_rasta( src_dataset_path=None, src_band=1,
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



