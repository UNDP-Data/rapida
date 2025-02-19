import io
import os.path
import logging
import pyarrow as pa
import geopandas
from exactextract import exact_extract
from typing import Iterable
from geopandas import GeoDataFrame
from osgeo import gdal, ogr, osr
from cbsurge.util.proj_are_equal import proj_are_equal
from osgeo_utils.gdal_calc import Calc

logger = logging.getLogger(__name__)
gdal.UseExceptions()
ogr.UseExceptions()

def geoarrow_schema_adapter(schema: pa.Schema, geom_field_name=None) -> pa.Schema:
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
    geometry_field_index = schema.get_field_index(geom_field_name)
    geometry_field = schema.field(geometry_field_index)
    geoarrow_geometry_field = geometry_field.with_metadata(
        {b"ARROW:extension:name": b"geoarrow.wkb"}
    )

    geoarrow_schema = schema.set(geometry_field_index, geoarrow_geometry_field)

    return geoarrow_schema

def calc(src_rasters:Iterable[str]=None, dst_raster=None, overwrite=False):
    assert dst_raster not in ('', None), f'Invalid dst_raster={dst_raster}'
    assert os.path.isabs(dst_raster), f'{dst_raster} is not an absolute path'
    target_srs = None
    files_to_sum = list()
    for n, src_raster in enumerate(src_rasters):
        prep_src_raster = None
        with gdal.OpenEx(src_raster) as src_rds:
            src_rast_srs = src_rds.GetSpatialRef()
            assert src_rast_srs is not None, f'Could not fetch SRS for {src_raster}'
            target_srs = src_rast_srs
            srs_are_equal = proj_are_equal(src_srs=target_srs, dst_srs=src_rast_srs)
            if not srs_are_equal:
                _, ext = os.path.splitext(src_raster)
                _, raster_fname = os.path.split(src_raster)
                prep_src_raster = f'/vsimem/{raster_fname.replace(ext, ".vrt")}'
                logger.info(f'Creating {prep_src_raster} VRT for {src_raster}')
                gdal.Warp(prep_src_raster, src_raster, format='VRT', dstSRS=target_srs,
                          creationOptions={'BLOCKXSIZE': 256, 'BLOCKYSIZE': 256})
            band = src_rds.GetRasterBand(1)
            block_size_x, block_size_y = band.GetBlockSize()
            # Get raster size
            width = src_rds.RasterXSize  # Number of columns
            height = src_rds.RasterYSize  # Number of rows

            intrs = set((height, width)).intersection((block_size_y, block_size_x))
            if intrs:
                _, ext = os.path.splitext(src_raster)
                _, raster_fname = os.path.split(src_raster)
                prep_src_raster = f'/vsimem/{raster_fname.replace(ext, ".vrt")}'
                logger.info(f'Creating {prep_src_raster} VRT for {src_raster}')
                gdal.Warp(prep_src_raster, src_raster, format='VRT',
                          creationOptions={'BLOCKXSIZE': 256, 'BLOCKYSIZE': 256})
            if prep_src_raster is None:
                prep_src_raster = src_raster
            files_to_sum.append(prep_src_raster)

    creation_options = 'TILED=YES COMPRESS=ZSTD BIGTIFF=IF_SAFER BLOCKXSIZE=256 BLOCKYSIZE=256 PREDICTOR=2'
    ds = Calc(calc='sum(a,axis=0)', a=files_to_sum, outfile=dst_raster, projectionCheck=True, format='GTiff',
              creation_options=creation_options.split(' '), quiet=True, overwrite=overwrite)


def sumup(src_rasters:Iterable[str]=None, dst_raster=None, overwrite=False) -> str:
    assert dst_raster not in ('', None), f'Invalid dst_raster={dst_raster}'
    assert os.path.isabs(dst_raster), f'{dst_raster} is not an absolute path'
    target_srs = None
    files_to_sum = list()
    for n, src_raster in enumerate(src_rasters):
        prep_src_raster = None
        with gdal.OpenEx(src_raster) as src_rds:
            src_rast_srs = src_rds.GetSpatialRef()
            assert src_rast_srs is not None, f'Could not fetch SRS for {src_raster}'
            target_srs = src_rast_srs
            srs_are_equal = proj_are_equal(src_srs=target_srs, dst_srs=src_rast_srs)
            if not srs_are_equal:
                _, ext = os.path.splitext(src_raster)
                _, raster_fname = os.path.split(src_raster)
                prep_src_raster = f'/vsimem/{raster_fname.replace(ext, ".vrt")}'
                logger.info(f'Creating {prep_src_raster} VRT for {src_raster}')
                gdal.Warp(prep_src_raster, src_raster, format='VRT', dstSRS=target_srs,
                          creationOptions={'BLOCKXSIZE': 256, 'BLOCKYSIZE': 256})
            band = src_rds.GetRasterBand(1)
            block_size_x, block_size_y = band.GetBlockSize()
            # Get raster size
            width = src_rds.RasterXSize  # Number of columns
            height = src_rds.RasterYSize  # Number of rows

            intrs = set((height,width)).intersection((block_size_y,block_size_x))
            if intrs:
                _, ext = os.path.splitext(src_raster)
                _, raster_fname = os.path.split(src_raster)
                prep_src_raster = f'/vsimem/{raster_fname.replace(ext, ".vrt")}'
                logger.info(f'Creating {prep_src_raster} VRT for {src_raster}')
                gdal.Warp(prep_src_raster, src_raster, format='VRT',
                          creationOptions={'BLOCKXSIZE': 256, 'BLOCKYSIZE': 256})
            if prep_src_raster is None:
                prep_src_raster = src_raster
            files_to_sum.append(prep_src_raster)
    creation_options = 'TILED=YES COMPRESS=ZSTD BIGTIFF=IF_SAFER BLOCKXSIZE=256 BLOCKYSIZE=256 PREDICTOR=2'
    ds = Calc(calc='sum(a,axis=0)', a=files_to_sum, outfile=dst_raster, projectionCheck=True, format='GTiff',
         creation_options=creation_options.split(' '), quiet=True, overwrite=overwrite)
    ds = None
    return dst_raster

def zonal_stats(src_rasters:Iterable[str] = None, polygon_ds=None, polygon_layer=None,
                vars_ops:Iterable[tuple[str, str]]=None, target_proj='ESRI:54009') -> GeoDataFrame:
    """
    Compute zonal stats form a set of raster files

    :param src_rasters: iterable of strings representing paths to the raster files
    :param polygon_ds: str representing the path to vector polygons
    :param polygon_layer, str, the name of the layert in polygon_ds used to zomute zonal stats
    :param vars_ops: a dict with same length as src_rasters where the first element represents the name of the column
    that will be created by applying the second element representing the operation
    :param target_proj: the projection represented as AUTHORITY:CODE, defaults to ESRI:54009 or Mollweide
    :return: pandas GeoDataFrame
    The function aligns the projections of all input data to the target_proj using in memory VRT's

    """
    target_srs = osr.SpatialReference()
    target_srs.SetFromUserInput(target_proj)

    with gdal.OpenEx(polygon_ds) as src_vds:
        lyr = src_vds.GetLayerByName(polygon_layer)
        src_vect_srs = lyr.GetSpatialRef()
        assert src_vect_srs is not None, f'Could not fetch SRS for {polygon_ds}'
        should_reproject_v = not proj_are_equal(src_srs=target_srs, dst_srs=src_vect_srs)
        if should_reproject_v:

            _, ext = os.path.splitext(polygon_ds)
            _, vector_fname = os.path.split(polygon_ds)
            prep_src_vector = f'/vsimem/{vector_fname.replace(ext, ".vrt")}'
            logger.info(f'Creating {prep_src_vector} VRT for {polygon_ds}')
            vrt_content = f"""<OGRVRTDataSource>
                <OGRVRTWarpedLayer>
                    <OGRVRTLayer name="{polygon_layer}">
                        <SrcDataSource>{polygon_ds}</SrcDataSource>
                        <SrcLayer>{polygon_layer}</SrcLayer>
                    </OGRVRTLayer>
                    <TargetSRS>{target_srs}</TargetSRS>
                </OGRVRTWarpedLayer>
            </OGRVRTDataSource>"""
            # geopandas can read from bytes buffer
            bio = io.BytesIO(vrt_content.encode('utf-8'))
            # exactextract can not, so we need a vsimem vrt
            gdal.FileFromMemBuffer(prep_src_vector,vrt_content)

        else:

            prep_src_vector = bio = polygon_ds
            
        combined_results = None
        for n, src_raster in enumerate(src_rasters):
            with gdal.OpenEx(src_raster) as src_rds:
                src_rast_srs = src_rds.GetSpatialRef()
                assert src_rast_srs is not None, f'Could not fetch SRS for {src_raster}'
                should_reproject_r = not proj_are_equal(src_srs=target_srs, dst_srs=src_rast_srs)
                if should_reproject_r:

                    _, ext = os.path.splitext(src_raster)
                    _, raster_fname = os.path.split(src_raster)
                    prep_src_raster = f'/vsimem/{raster_fname.replace(ext, ".vrt")}'
                    logger.debug(f'Creating {prep_src_raster} VRT for {src_raster}')
                    gdal.Warp(prep_src_raster, src_raster, format='VRT', dstSRS=target_srs,creationOptions={'BLOCKXSIZE':256, 'BLOCKYSIZE':256} )
                    # assert os.path.exists(prep_src_raster)
                else:
                    prep_src_raster = src_raster
                var_name, operation = vars_ops[n]
                df = exact_extract(prep_src_raster, lyr, ops=[operation], include_geom=not n, output='pandas')
                df.rename(columns={operation: var_name}, inplace=True)

                df['tempid'] = range(1, len(df) + 1)
                if not n:
                    combined_results = df
                else:
                    combined_results = combined_results.merge(df, on=['tempid'], how='inner')

                if should_reproject_r:gdal.Unlink(prep_src_raster)
        egdf = geopandas.read_file(bio, layer=polygon_layer)
        for e in vars_ops:
            vname = e[0]
            if vname in egdf.columns.tolist():
                egdf.drop(columns=[vname], inplace=True)
        combined_results = egdf.merge(df, on='geometry',how='inner')

        if should_reproject_v:
            gdal.Unlink(prep_src_vector)

            assert os.path.exists(prep_src_vector)
        if isinstance(bio, io.BytesIO):bio.close()
        combined_results.drop(columns='tempid', inplace=True)

        return combined_results
