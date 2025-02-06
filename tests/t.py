import geopandas
import click
polygons = '/data/adhoc/MDA/adm/adm3transn.fgb'
import os
from pyproj import CRS
# with gdal.OpenEx(polygons) as poly_ds:
                #     lcount = poly_ds.GetLayerCount()
                #     if lcount > 1:
                #         lnames = list()
                #         for i in range(lcount):
                #             l = poly_ds.GetLayer(i)
                #             lnames.append(l.GetName())
                #         #click.echo(f'{polygons} contains {lcount} layers: {",".join(lnames)}')
                #         layer_name = click.prompt(
                #             f'{polygons} contains {lcount} layers: {",".join(lnames)} Please type/select  one or pres enter to skip if you wish to use default value',
                #             type=str, default=lnames[0])
                #     else:
                #         layer_name = poly_ds.GetLayer(0).GetName()
                #     if not os.path.exists(self.data_folder):
                #         os.makedirs(self.data_folder)
                #     gdal.VectorTranslate(self.geopackage_file_path, poly_ds, format='GPKG',reproject=True, dstSRS=projection,
                #                      layers=[layer_name], layerName='polygons', geometryType='PROMOTE_TO_MULTI', makeValid=True)
                #


l = geopandas.list_layers(polygons)
lnames = l.name.tolist()
lcount = len(lnames)
if lcount > 1:
    click.echo(f'{polygons} contains {lcount} layers: {",".join(lnames)}')
    layer_name = click.prompt(
        f'{polygons} contains {lcount} layers: {",".join(lnames)} Please type/select  one or pres enter to skip if you wish to use default value',
        type=str, default=lnames[0])
else:
    layer_name = lnames[0]

# if not os.path.exists(self.data_folder):
#     os.makedirs(self.data_folder)
#

gdf = geopandas.read_file(polygons, layer=layer_name,)
rgdf = gdf.to_crs(crs=CRS.from_user_input('ESRI:54009'))
rgdf.to_file(filename='/tmp/a.gpkg', driver='GPKG', engine='pyogrio', mode='w', layer='polygons', promote_to_multi=True)