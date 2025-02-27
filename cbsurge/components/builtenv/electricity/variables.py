import json
import os
from collections import OrderedDict
import logging

from cbsurge.util.setup_logger import setup_logger

logger = logging.getLogger(__name__)

def generate_variables():
    """
    Generate electricity variables dict
    :return:
    """

    variables = OrderedDict()

    #dependencies
    variables['electricity_grid_length'] = dict(
        title='Total length of electricity grid',
        source='geohub:/api/datasets/310aadaa61ea23811e6ecd75905aaf29')
    return variables


if __name__ == '__main__':
    logger = setup_logger(name='rapida')

    variables = generate_variables()
    #print(json.dumps(variables, indent=2))
    from cbsurge.components.builtenv.electricity import ElectricityVariable, ElectricityComponent
    from geopandas import list_layers
    from osgeo import gdal, osr
    dst_name = '../../../../ap/data/ap.gpkg'
    l = list_layers(dst_name)

    print(ElectricityComponent.dataset_url)
    url = 'https://undpgeohub.blob.core.windows.net/userdata/4c9b6906ff63494418aef1efd028be44/datasets/grid_20250217093237.gpkg/grid.pmtiles.fgb?sv=2025-01-05&ss=b&srt=o&se=2026-02-21T14%3A08%3A11Z&sp=r&sig=kZyLVLYbRUn5oGbGGudBQGSf97Kh7SlRX2l0uMcGSo4%3D'


    ElectricityVariable.download_geodata_by_admin(dataset_url=url,
                                                  geopackage_path=dst_name,)


    # import geopandas
    # lurl = '../../../../ap/data/grid.gpkg'
    # nakuru = '../../../../ap/data/nakuru.fgb'
    #
    # gdf = geopandas.read_file(nakuru)
    # bb = gdf.total_bounds
    # print(bb)
    # print(gdf.geometry.iloc[0])
    # path = os.path.abspath(dst_name)
    # assert  os.path.exists(path)
    # dst_srs = osr.SpatialReference()
    # dst_srs.ImportFromEPSG(4326)
    # dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    #
    # with gdal.OpenEx(path) as src:
    #     l = src.GetLayerByName('polygons')
    #     admin_srs = l.GetSpatialRef()
    #     print(admin_srs.GetDataAxisToSRSAxisMapping())
    #     print(dst_srs.GetDataAxisToSRSAxisMapping())
    #     l.SetAttributeFilter('name = "Nakuru"')
    #     for f in l:
    #        geom = f.GetGeometryRef()
    #        geom.TransformTo(dst_srs)
    #        print(geom)
    #
    # with gdal.OpenEx(path) as src:
    #     l = src.GetLayerByName('polygons_4326')
    #     admin_srs = l.GetSpatialRef()
    #     l.SetAttributeFilter('name = "Nakuru"')
    #     for f in l:
    #        geom = f.GetGeometryRef()
    #        # geom.TransformTo(dst_srs)
    #        print(geom)
    # # with gdal.OpenEx(path) as src:
    # #     l = src.GetLayerByName('polygons_54034')
    # #     admin_srs = l.GetSpatialRef()
    # #     l.SetAttributeFilter('name = "Nakuru"')
    # #     for f in l:
    # #        geom = f.GetGeometryRef()
    # #        geom.TransformTo(dst_srs)
    # #        print(geom.Boundary().GetEnvelope())
