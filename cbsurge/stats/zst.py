import os.path
import logging
from exactextract import exact_extract
from typing import Iterable
from geopandas import GeoDataFrame, read_file
from osgeo import gdal, ogr, osr
from cbsurge.util import proj_are_equal

logger = logging.getLogger(__name__)
gdal.UseExceptions()




def zonal_stats(src_rasters:Iterable[str] = None, src_vector=None, vars_ops=None,  target_proj='ESRI:54034'):
    target_srs = osr.SpatialReference()
    target_srs.SetFromUserInput(target_proj)



    with gdal.OpenEx(src_vector) as src_vds:
        lyr = src_vds.GetLayer(0)
        layer_name = lyr.GetName()
        src_vect_srs = lyr.GetSpatialRef()
        assert src_vect_srs is not None, f'Could not fetch SRS for {src_vector}'
        srs_are_equal = proj_are_equal(src_srs=target_srs, dst_srs=src_vect_srs)
        if not srs_are_equal:
            logger.info(f'Creating VRT for {src_vector}')
            _, ext = os.path.splitext(src_vector)
            prep_src_vector = src_vector.replace(ext, '.vrt')
            with open(prep_src_vector, 'wt') as v_vrt:
                vrt_content = f"""<OGRVRTDataSource>
                    <OGRVRTWarpedLayer>
                        <OGRVRTLayer name="{layer_name}">
                            <SrcDataSource>{src_vector}</SrcDataSource>
                            <SrcLayer>{layer_name}</SrcLayer>
                        </OGRVRTLayer>
                        <TargetSRS>{target_srs}</TargetSRS>
                    </OGRVRTWarpedLayer>
                </OGRVRTDataSource>"""
                v_vrt.write(vrt_content)

            assert os.path.exists(prep_src_vector)
        else:
            prep_src_vector = src_vector
        combined_results = None
        for n, src_raster in enumerate(src_rasters):
            with gdal.OpenEx(src_raster) as src_rds:
                src_rast_srs = src_rds.GetSpatialRef()
                assert src_rast_srs is not None, f'Could not fetch SRS for {src_raster}'
                srs_are_equal = proj_are_equal(src_srs=target_srs, dst_srs=src_rast_srs)
                if not srs_are_equal:
                    logger.info(f'Creating VRT for {src_raster}')
                    _, ext = os.path.splitext(src_raster)
                    prep_src_raster = src_raster.replace(ext, '.vrt')
                    gdal.Warp(prep_src_raster, src_raster, format='VRT', dstSRS=target_srs,creationOptions={'BLOCKXSIZE':256, 'BLOCKYSIZE':256} )
                    assert os.path.exists(prep_src_raster)
                else:
                    prep_src_raster = src_raster
                var_name, operation = vars_ops[n]
                df = exact_extract(prep_src_raster, prep_src_vector, ops=[operation], include_geom=not n, output='pandas')
                df.rename(columns={operation: var_name}, inplace=True)

                df['tempid'] = range(1, len(df) + 1)
                if not n:
                    combined_results = df
                else:
                    combined_results = combined_results.merge(df, on=['tempid'], how='inner')
        combined_results = combined_results.merge(read_file(prep_src_vector, engine='pyogrio'), on='geometry',
                                                  how='inner')
        combined_results.drop(columns='tempid', inplace=True)
        return combined_results
