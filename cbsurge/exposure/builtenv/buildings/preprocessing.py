import logging
import multiprocessing
import random
from multiprocessing import sharedctypes
import threading
import numpy as np
import pyarrow as pa
import pyproj
import shapely
from osgeo import gdal, ogr, osr


from cbsurge import util
from collections import deque
from rich.progress import Progress
import concurrent
from pyogrio import read_info, read_arrow, read_dataframe
from rasterio import warp
# from pyogrio.raw import open_arrow, read_arrow
import rasterio
from rasterio.windows import Window
from pyarrow import compute as pc
from cbsurge.constants import ARROWTYPE2OGRTYPE
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

def proj_are_equal(src_srs: osr.SpatialReference = None, dst_srs: osr.SpatialReference = None):
    """
    Decides if two projections are equal
    @param src_srs:  the source projection
    @param dst_srs: the dst projection
    @return: bool, True if the source  is different then dst else false
    If the src is ESPG:4326 or EPSG:3857  returns  False
    """
    auth_code_func_name = ".".join(
        [osr.SpatialReference.GetAuthorityCode.__module__, osr.SpatialReference.GetAuthorityCode.__name__])
    is_same_func_name = ".".join([osr.SpatialReference.IsSame.__module__, osr.SpatialReference.IsSame.__name__])
    try:
        proj_are_equal = int(src_srs.GetAuthorityCode(None)) == int(dst_srs.GetAuthorityCode(None))
    except Exception as evpe:
        logger.error(
            f'Failed to compare src and dst projections using {auth_code_func_name}. Trying using {is_same_func_name}')
        try:
            proj_are_equal = bool(src_srs.IsSame(dst_srs))
        except Exception as evpe1:
            logger.error(
                f'Failed to compare src and dst projections using {is_same_func_name}. Error is \n {evpe1}')
            raise evpe1

    return proj_are_equal

def filter_buildings_in_block(buildings_ds_path=None, mask_ds=None, block=None, block_id=None, band=1):

    try:

        window = Window(*block)
        col_start, row_start, col_size, row_size = block
        m = mask_ds.read(band, window=window)

        bbox = rasterio.windows.bounds(window=window, transform=mask_ds.transform)

        ds = read_dataframe(buildings_ds_path, bbox=bbox, read_geometry=True)
        ds.rename_geometry('wkb_geometry')

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
        table = table.rename_columns(names={'geometry':'wkb_geometry'})

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
            blocks = util.gen_blocks(blockxsize=block_xsize, blockysize=block_ysize, width=width, height=height)
            nblocks, blocks = util.generator_length(blocks)
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

if __name__ == '__main__':
    import time

    logger = util.setup_logger(name='rapida', level=logging.INFO, make_root=True)


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