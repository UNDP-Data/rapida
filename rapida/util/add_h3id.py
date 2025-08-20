import h3.api.basic_int as h3
from osgeo import gdal, ogr


def add_h3id(dataset_path=None, h3id_precision=7, layer_name=None, h3id_column=None):
    with gdal.OpenEx(dataset_path, gdal.OF_VECTOR|gdal.OF_UPDATE) as ds:

        layer = ds.GetLayerByName(layer_name) if isinstance(layer_name, str) else ds.GetLayer(layer_name)
        h3id_field_index = layer.GetLayerDefn().GetFieldIndex(h3id_column)
        if h3id_field_index < 0:
            h3id_field = ogr.FieldDefn(h3id_column, ogr.OFTInteger64)
            layer.CreateField(h3id_field)
            for feature in layer:
                geom = feature.GetGeometryRef()
                centroid = geom.Centroid()
                h3id = h3.latlng_to_cell(lat=centroid.GetY(), lng=centroid.GetX(),
                                         res=h3id_precision)
                feature.SetField(h3id_column, h3id)
                layer.SetFeature(feature)
        else:
            raise Exception(f'{layer} already has h3id field!, Please delete it first! \n'
                            f'ogrinfo {dataset_path} -sql "ALTER TABLE {layer_name} DROP COLUMN {h3id_column}"')