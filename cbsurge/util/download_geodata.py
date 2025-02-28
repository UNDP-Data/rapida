import concurrent.futures
import logging
import random
import threading
import time
from collections import deque
import tempfile
import numpy
import pyarrow as pa
import shapely
from osgeo import gdal, ogr, osr
from pyogrio import read_info
from pyproj import Transformer
from rich.progress import Progress
from shapely import wkb
from shapely.ops import transform, unary_union
from shapely.geometry import box
import os
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd

from cbsurge import constants
from cbsurge.constants import ARROWTYPE2OGRTYPE
from cbsurge.util.downloader import downloader


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


def download_raster(
        dataset_url: str,
        geopackage_path: str,
        output_filename: str,
        mask_layer_name: str = None,
        progress: Progress=None,
        chunk_size: tuple[int, int]=(4096, 4096),
):
    """
    download raster data clipped by the project area

    :param dataset_url: raster dataset url
    :param geopackage_path: project geopackage path. It should contain `polygons` layer unless set layer name as mask_layer_name
    :param output_filename: output filename to save the downloaded raster file
    :param mask_layer_name: mask layer name. If not specified, it clips the raster by `polygons` layer.
    :param progress: rich progress instance
    :return: output file path is returned
    """
    download_task = None
    if progress:
        download_task = progress.add_task(
            description=f'[red]Going to download data covering the project area', total=None)

    output_dir = os.path.dirname(geopackage_path)

    mask_layer = constants.POLYGONS_LAYER_NAME
    if mask_layer_name:
        mask_layer = mask_layer_name
    gdf = gpd.read_file(geopackage_path, layer=mask_layer)
    polygons_crs = gdf.crs

    with rasterio.open(dataset_url) as src:
        # Reproject the polygons to match the raster CRS
        gdf_reproj = gdf.to_crs(src.crs)
        geoms = gdf_reproj.geometry.values.tolist()

        if progress and download_task is not None:
            progress.update(download_task, description=f'[red] Identified {len(geoms)} polygon(s) for masking.')

        # Create a single unioned geometry from all polygons for efficiency
        union_geom = unary_union(geoms)
        # Compute the bounding box of the unioned polygon
        minx, miny, maxx, maxy = union_geom.bounds

        # Compute the overall window in pixel coordinates that covers the unioned bounds
        overall_window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)

        # Set chunk size in pixels (you can adjust these values as needed)
        chunk_width = chunk_size[0]
        chunk_height = chunk_size[1]

        temp_files = []  # List to hold temporary file paths for each chunk

        # Create a temporary directory for storing chunk files
        with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
            # Get integer bounds of the overall window
            col_off = int(overall_window.col_off)
            row_off = int(overall_window.row_off)
            win_width = int(overall_window.width)
            win_height = int(overall_window.height)

            # Loop over the overall window in chunks
            rows = range(row_off, row_off + win_height, chunk_height)
            cols = range(col_off, col_off + win_width, chunk_width)

            total_chunks = len(rows) * len(cols)
            if progress and download_task is not None:
                progress.update(download_task, description=f'[red] Downloading {total_chunks} chunks')

            for row in rows:
                for col in cols:
                    # Define a window (chunk) with proper size limits
                    w = rasterio.windows.Window(
                        col, row,
                        min(chunk_width, col_off + win_width - col),
                        min(chunk_height, row_off + win_height - row)
                    )
                    # Compute the spatial bounds of the window
                    w_bounds = rasterio.windows.bounds(w, transform=src.transform)
                    # Create a shapely box for the window bounds
                    window_box = box(*w_bounds)
                    # Skip this window if it does not intersect the unioned polygon
                    if not window_box.intersects(union_geom):
                        continue

                    # Read the chunk data from the source raster
                    data = src.read(window=w)
                    # Get the transform for this window
                    w_transform = src.window_transform(w)

                    # Create a boolean mask for pixels inside the polygon(s)
                    mask_arr = rasterio.features.geometry_mask(
                        geoms,
                        out_shape=(int(w.height), int(w.width)),
                        transform=w_transform,
                        invert=True
                    )
                    # Determine nodata value; if not set, default to 0
                    nodata = src.nodata if src.nodata is not None else 0
                    # Apply the mask: set pixels outside the polygon(s) to nodata
                    data = numpy.where(mask_arr, data, nodata)

                    # Prepare metadata for the chunk
                    chunk_meta = src.meta.copy()
                    chunk_meta.update({
                        "height": int(w.height),
                        "width": int(w.width),
                        "transform": w_transform
                    })

                    # Write the chunk to a temporary file
                    chunk_filename = os.path.join(tmpdir, f"chunk_{row}_{col}.tif")
                    with rasterio.open(chunk_filename, "w", **chunk_meta) as dst:
                        dst.write(data)
                    temp_files.append(chunk_filename)

                    if progress and download_task is not None:
                        progress.update(download_task,
                                        description=f'[red] Downloaded chunk at row {row}, col {col}: ({len(temp_files)} / {total_chunks})')

            # Merge all chunk files into one mosaic
            src_files_to_mosaic = []
            for fp in temp_files:
                src_ds = rasterio.open(fp)
                src_files_to_mosaic.append(src_ds)

            mosaic, mosaic_transform = merge(src_files_to_mosaic)

            # Update metadata based on the merged mosaic
            out_meta = src.meta.copy()
            out_meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": mosaic_transform
            })

            # Close all temporary datasets
            for src_ds in src_files_to_mosaic:
                src_ds.close()

        if progress and download_task is not None:
            progress.update(download_task, description=f'[red] Merged all chunks; starting reprojection from {out_meta["crs"]} to {polygons_crs.name}.')

        # Compute the spatial bounds of the merged mosaic
        mosaic_bounds = rasterio.transform.array_bounds(mosaic.shape[1], mosaic.shape[2], mosaic_transform)

        # Calculate the destination transform and dimensions for the target CRS
        dst_transform, dst_width, dst_height = calculate_default_transform(
            out_meta["crs"],
            polygons_crs,
            mosaic.shape[2],
            mosaic.shape[1],
            *mosaic_bounds
        )

        # Reproject the mosaic to the polygon CRS
        dst_image = numpy.empty((mosaic.shape[0], dst_height, dst_width), dtype=mosaic.dtype)
        for i in range(mosaic.shape[0]):
            reproject(
                source=mosaic[i],
                destination=dst_image[i],
                src_transform=mosaic_transform,
                src_crs=out_meta["crs"],
                dst_transform=dst_transform,
                dst_crs=polygons_crs,
                resampling=Resampling.nearest
            )

        # Update metadata for the reprojected raster
        out_meta.update({
            "driver": "COG",
            "height": dst_height,
            "width": dst_width,
            "transform": dst_transform,
            "crs": polygons_crs,
            "compress": 'zstd',
            "predictor": 2
        })

        if progress and download_task is not None:
            progress.update(download_task, description=f'[red] Storing transformed data.')

    output_path = os.path.join(output_dir, output_filename)
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(dst_image)
        if progress is not None and download_task is not None:
            progress.update(download_task, description=f'[red] stored data at {output_path} successfully.')

    if progress and download_task:
        progress.remove_task(download_task)

    return output_path
