import fiona
import geopandas as gpd


def gdf_columns(file_path, layer_name):
    with fiona.open(fp=file_path, layer=layer_name, mode='r') as src:
        schema = src.schema
        fields = schema['properties'].keys()
        return fields

def gdf_chunks(file_path, chunk_size, layer_name):
    with fiona.open(fp=file_path, layer=layer_name, mode='r') as src:
        crs = src.crs
        chunk = []
        for i, feature in enumerate(src):
            chunk.append(feature)
            if (i + 1) % chunk_size == 0:
                gdf = gpd.GeoDataFrame.from_features(chunk, crs=crs)
                # Process the chunk
                yield gdf
                chunk = []

        # Process remaining
        if chunk:
            gdf = gpd.GeoDataFrame.from_features(chunk, crs=crs)
            yield gdf

