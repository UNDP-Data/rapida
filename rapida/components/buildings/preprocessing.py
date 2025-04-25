import concurrent
import logging
import os.path
import random
import threading
from collections import deque
import time
import numpy as np
import pyarrow as pa
# from pyogrio.raw import open_arrow, read_arrow
import rasterio
from osgeo import gdal, ogr, osr
from osgeo.ogr import wkbPolygon
from pyogrio import read_dataframe, open_arrow
from rasterio.windows import Window
from rich.progress import Progress
from rapida.util.worker import worker as genworker
from rapida.constants import ARROWTYPE2OGRTYPE
from rapida.util.gen_blocks import gen_blocks
from rapida.util.generator_length import generator_length
from rapida.util.setup_logger import setup_logger
from rich.progress import TimeElapsedColumn
logger = logging.getLogger(__name__)
gdal.UseExceptions()

def cb(complete, message, stop_event):
    #logger.info(f'{complete * 100:.2f}%')
    if stop_event and stop_event.is_set():
        logger.info(f'GDAL was signalled to stop')
        return 0

def create_centroid(src_path=None, src_layer_name=None, dst_path=None, dst_srs=None):
    """
    Compute centroid for buildings layer
    :param src_path: str, path to input polygons
    :param src_layer_name, str, the layer name
    :param dst_path: str, path to out centroids
    :return: None
    """
    assert src_layer_name not in ('', None), f'invalid src_layer_name={src_layer_name}'

    options = dict(
        format='FlatGeobuf',
        SQLDialect='sqlite',
        SQLStatement=f'SELECT ST_PointOnSurface(geometry) as geometry, rowid as fid FROM {src_layer_name}',
        layerName=src_layer_name,
        geometryType='POINT'

    )

    if not ogr.GetDriverByName(options['format']).TestCapability(ogr.OLCFastFeatureCount):
        logger.debug('No progress bar is available in create_centroid')

    if dst_srs is not None:
        options['dstSRS'] = dst_srs
        options['reproject'] = True

    options = gdal.VectorTranslateOptions(**options)


    ds = gdal.VectorTranslate(destNameOrDestDS=dst_path,srcDS=src_path,options=options)

    del ds


def geoarrow_schema_adapter(schema: pa.Schema) -> pa.Schema:
    """
    Convert a geoarrow-compatible schema to a proper geoarrow schema

    This assumes there is a single "geometry" column with WKB formatting

    Parameters
    ----------
    schema: pa.Schema

    Returns
    -------
    pa.Schema
    A copy of the input schema with the geometry field replaced with
    a new one with the proper geoarrow ARROW:extension metadata

    """
    geometry_field_index = schema.get_field_index("geometry")
    geometry_field = schema.field(geometry_field_index)
    geoarrow_geometry_field = geometry_field.with_metadata(
        {b"ARROW:extension:name": b"geoarrow.wkb"}
    )

    geoarrow_schema = schema.set(geometry_field_index, geoarrow_geometry_field)

    return geoarrow_schema
from pyogrio.geopandas import read_dataframe

def mask_buildings_in_block(buildings_dataset=None, buildings_layer_name=None,
                            mask_ds=None, block=None, name=None, results=None, progress=None, band=1):

    try:
        if progress:
            task = progress.add_task(description=f'[green]Filtering buildings data in {name}...', start=False,
                                     total=None, )

        window = Window(*block)
        col_start, row_start, col_size, row_size = block
        ds_mask = mask_ds.read(band, window=window)

        bbox = rasterio.windows.bounds(window=window, transform=mask_ds.transform)
        buildings_gdf = read_dataframe(buildings_dataset, layer=buildings_layer_name,  bbox=bbox, read_geometry=True)
        if len(buildings_gdf) == 0:
            logger.debug(f'No buildings exist in {name}-{bbox}')
            return name
        buildings_centroids = buildings_gdf.centroid.get_coordinates()
        point_cols, point_rows = ~mask_ds.transform * (buildings_centroids.x, buildings_centroids.y)
        point_rows, point_cols = np.floor(point_rows).astype('i4'), np.floor(point_cols).astype('i4')
        point_rows -= row_start
        point_cols -= col_start
        row_mask = (point_rows >= 0) & (point_rows < row_size)
        col_mask = (point_cols >= 0) & (point_cols < col_size)
        rc_mask = row_mask & col_mask
        if rc_mask[rc_mask].size == 0:
            logger.debug(f'No buildings exist in {name}-{bbox} after applying mask')
            return name
        point_rows = point_rows[rc_mask]
        point_cols = point_cols[rc_mask]
        buildings_gdf = buildings_gdf[rc_mask]
        block_mask = ds_mask[point_rows, point_cols] == True
        masked_buildings_gdf = buildings_gdf[block_mask]
        if masked_buildings_gdf.size > 0:
            arrow_object = masked_buildings_gdf.to_arrow(index=False)
            table = pa.table(arrow_object)
            #table = table.rename_columns(names={'geometry': 'wkb_geometry'})
            results.append((name, table))
        return name
    except Exception as e:
        raise
    finally:
        if progress:
            progress.remove_task(task)



def filter_buildings_in_block(buildings_ds_path=None,  mask_ds=None,
                            block=None, block_id=None, band=1):

    try:

        window = Window(*block)
        col_start, row_start, col_size, row_size = block
        m = mask_ds.read(band, window=window)

        bbox = rasterio.windows.bounds(window=window, transform=mask_ds.transform)

        ds = read_dataframe(buildings_ds_path, bbox=bbox, read_geometry=True)

        if len(ds) == 0:
            return block_id, None, None
        pcoords = ds.centroid.get_coordinates()
        pcols, prows = ~mask_ds.transform * (pcoords.x, pcoords.y)
        prows, pcols = np.floor(prows).astype('i4'), np.floor(pcols).astype('i4')
        prows -= row_start
        pcols -= col_start
        rowmask = (prows >= 0) & (prows < row_size)
        colmask = (pcols >= 0) & (pcols < col_size)
        rcmask = rowmask & colmask
        if rcmask[rcmask].size == 0:
            return block_id, None, None
        prows = prows[rcmask]
        pcols = pcols[rcmask]
        ds = ds[rcmask]
        rm = m[prows, pcols] == True
        mds = ds[rm]
        ao = mds.to_arrow(index=False)
        table = pa.table(ao)

        # schema = geoarrow_schema_adapter(table.schema)
        # table = pa.table(table, schema=schema)
        #table = table.rename_columns(names={'geometry':'wkb_geometry'})

        if mds.size == 0:
            return  block_id, None, None
        out_srs = osr.SpatialReference()
        out_srs.SetFromUserInput(':'.join(mds.crs.to_authority()))
        return block_id, table, out_srs
    except Exception as e:

        return block_id, None, None



def worker(work=None, result=None, finished=None):


    logger.debug(f'starting building filter thread {threading.current_thread().name}')
    while True:

        job = None
        try:
            job = work.pop()
        except IndexError as ie:
            pass
        if job is None:
            if finished.is_set():
                logger.debug(f'worker is finishing  in {threading.current_thread().name}')
                break
            continue

        if finished.is_set():
            break
        logger.debug(f'Starting job in block {job["block_id"]}')

        result.append(filter_buildings_in_block(**job))



def filter_buildings(buildings_path=None, mask_path=None, mask_pixel_value=None,
                     horizontal_chunks=None, vertical_chunks=None, nworkers=1,
                     out_path=None):
    """
    Select buildings whose centroid is inside the masked pixels
    :param buildings_path:
    :param mask_path:
    :param mask_pixel_value:
    :param horizontal_chunks:
    :param vertical_chunks:
    :return:
    """

    nfiltered = 0
    failed = []


    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
        with rasterio.open(mask_path) as mask_ds:
            msr = osr.SpatialReference()
            msr.SetFromUserInput(str(mask_ds.crs))
            # should_reproj = not proj_are_equal(src_srs=bsr, dst_srs=msr)
            # assert should_reproj is False, f'{buildings_path} and {mask_path} need to be in the same projection'
            width = mask_ds.width
            height = mask_ds.height
            assert mask_ds.count == 1, f'The mask dataset {mask_path} contains more than one band'
            #mband = mds.GetRasterBand(1)
            #m = mband.ReadAsArray()
            #ctypes_shared_mem = sharedctypes.RawArray(np.ctypeslib.as_ctypes_type(m.dtype), m.ravel())
            # width = mds.RasterXSize
            # height = mds.RasterYSize
            block_xsize = width // horizontal_chunks
            block_ysize = height // vertical_chunks
            blocks = gen_blocks(blockxsize=block_xsize, blockysize=block_ysize, width=width, height=height)
            nblocks, blocks = generator_length(blocks)
            stop_event = threading.Event()
            jobs = deque()
            results = deque()
            nworkers = nblocks if nworkers > nblocks else nworkers
            print(nworkers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
                [executor.submit(worker, jobs, results, stop_event) for i in range(nworkers)]
                with Progress() as progress:
                    total_task = progress.add_task(
                        description=f'[red]Going to filter buildings from {nblocks} blocks/chunks',
                        total=nblocks)
                    for block_id, block in enumerate(blocks):
                        job = dict(
                            buildings_ds_path=buildings_path,
                            mask_ds=mask_ds,
                            block=block,
                            block_id=block_id,

                        )
                        jobs.append(job)


                    while True:
                        try:
                            try:
                                block_id, table, dst_srs = results.pop()

                                if table is None:
                                    nfiltered += 1
                                    progress.update(total_task,
                                                    description=f'[red]Filtered buildings from {nfiltered} out of {nblocks} blocks',
                                                    advance=1)
                                    continue

                                if dst_ds.GetLayerCount() == 0:

                                    dst_lyr = dst_ds.CreateLayer('buildings_filtered', geom_type=ogr.wkbPolygon, srs=dst_srs,
                                                                 )
                                    for name in table.schema.names:
                                        if 'wkb' in name or 'geometry' in name: continue

                                        field = table.schema.field(name)
                                        field_type = ARROWTYPE2OGRTYPE[field.type]
                                        logger.debug(f'Creating field {name}: {field.type}: {field_type}')
                                        dst_lyr.CreateField(ogr.FieldDefn(name, field_type))


                                try:

                                    dst_lyr.WritePyArrow(table)
                                except Exception as e:
                                    logger.error(
                                        f'Failed to write {table.num_rows} features/rows in block id {block_id} because {e}. Skipping')

                                dst_lyr.SyncToDisk()
                                nfiltered += 1
                                progress.update(total_task,
                                                description=f'[red]Filtered buildings from {nfiltered} out of {nblocks} blocks',
                                                advance=1)
                                logger.debug(f'{block_id} was processed')
                            except IndexError as ie:
                                if not jobs and progress.finished:
                                    stop_event.set()
                                    break
                                s = random.random()  # this one is necessary for ^C/KeyboardInterrupt
                                time.sleep(s)

                                continue
                        except Exception as e:
                            failed.append(f'Error in block_id {block_id} failed: {e.__class__.__name__}("{e}")')
                            progress.update(total_task,
                                            description=f'[red]Filtered  buildings from {nfiltered} out of {nblocks} blocks',
                                            advance=1)
                            nfiltered+=1

                        except KeyboardInterrupt:
                            logger.info(f'Cancelling jobs. Please wait/allow for a graceful shutdown')
                            stop_event.set()
                            break
        logger.info(f'{dst_lyr.GetFeatureCount()} feature were written to {out_path} ')
    if failed:
        for msg in failed:
            logger.error(msg)


def mask_buildings( buildings_dataset=None, buildings_layer_name=None,mask_ds_path=None,
                    masked_buildings_dataset=None, masked_buildings_layer_name=None,
                    horizontal_chunks=None, vertical_chunks=None, workers=4, progress=None
                   ):
    """
    Filter buildings based on a raster mask
    :param buildings_dataset:
    :param buildings_layer_name:
    :param mask_ds_path:
    :param masked_buildings_dataset:
    :param masked_buildings_layer_name:
    :param horizontal_chunks:
    :param vertical_chunks:
    :param workers:
    :param progress:
    :return:
    """
    total_task = None
    if progress:
        cols = progress.columns
        progress.columns = [e for e in cols] + [TimeElapsedColumn()]
    try:

        with rasterio.open(mask_ds_path) as mask_ds:
            assert mask_ds.count == 1, f'The mask dataset {mask_ds_path} contains more than one band'
            width = mask_ds.width
            height = mask_ds.height
            block_xsize = width // horizontal_chunks
            block_ysize = height // vertical_chunks
            blocks = gen_blocks(blockxsize=block_xsize, blockysize=block_ysize, width=width, height=height)
            nblocks, blocks = generator_length(blocks)
            stop_event = threading.Event()
            jobs = deque()
            results = deque()
            workers = nblocks if workers > nblocks else workers
            folder_path, _ = os.path.split(masked_buildings_dataset)
            out_path = os.path.join(folder_path, f'{masked_buildings_layer_name}.fgb')
            with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
                with gdal.OpenEx(buildings_dataset, gdal.OF_VECTOR) as bldgs_ds:
                    bldgs_layer = bldgs_ds.GetLayerByName(buildings_layer_name)
                    bldgs_layer_defn = bldgs_layer.GetLayerDefn()

                    destination_layer = dst_ds.CreateLayer(
                        masked_buildings_layer_name,
                        srs=bldgs_layer.GetSpatialRef(),
                        geom_type=bldgs_layer_defn.GetGeomType(),
                    )
                    # Copy fields
                    for i in range(bldgs_layer_defn.GetFieldCount()):
                        field_defn = bldgs_layer_defn.GetFieldDefn(i)
                        destination_layer.CreateField(field_defn)
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:

                    if progress:
                        total_task = progress.add_task(
                            description=f'[red]Going to mask buildings in {nblocks} blocks/chunks',
                            total=nblocks)
                    for block_id, block in enumerate(blocks):
                        job = dict(
                            buildings_dataset=buildings_dataset,
                            buildings_layer_name=buildings_layer_name,
                            mask_ds=mask_ds,
                            block=block,
                            name=f'block::{block_id}',
                            results=results,
                            progress=progress

                        )

                        jobs.append(job)
                    futures = [executor.submit(genworker, job=mask_buildings_in_block, jobs=jobs, stop=stop_event, task=total_task, id_prop_name='name') for i in range(workers)]
                    while True:
                        try:
                            try:
                                block_id, table = results.pop()
                                table = table.rename_columns({'geometry':'wkb_geometry'})
                                try:
                                    destination_layer.WritePyArrow(table)
                                    destination_layer.SyncToDisk()
                                except Exception as e:
                                    logger.error(
                                        f'Failed to write {table.num_rows} features/rows in block id {block_id} because {e}. Skipping')

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

            #with gdal.config_option(key="OGR2OGR_USE_ARROW_API", value="NO"):
                # ogr Arrow is used in translate > 3.8 and it has some issue with OGC_FID col
            with gdal.VectorTranslate(destNameOrDestDS=masked_buildings_dataset,srcDS=out_path, accessMode='append') as ds:
                pass
            os.remove(out_path)
    except Exception as e:
        logger.error(f'Error masking {buildings_dataset} with error {e}')
        raise

    finally:
        if progress and total_task:
            progress.remove_task(total_task)
            progress.columns = cols
if __name__ == '__main__':
    import time

    logger = setup_logger(name='rapida', level=logging.INFO, make_root=True)


    src_path = '/data/surge/buildings_eqar.fgb'
    dst_path = '/data/surge/bldgs1c.fgb'
    mask = '/data/surge/surge/stats/floods_mask.tif'
    filtered_buildings_path = '/data/surge/bldgs1_filtered.fgb'

    start = time.time()

    #create_centroid(src_path=src_path, src_layer_name='buildings', dst_path=dst_path, dst_srs='ESRI:54034')
    filter_buildings(
        buildings_path=src_path,
        mask_path=mask,
        mask_pixel_value=1,
        horizontal_chunks=10,
        vertical_chunks=20,
        out_path=filtered_buildings_path
    )

    end = time.time()
    print((end-start))