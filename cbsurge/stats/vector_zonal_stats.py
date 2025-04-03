import logging
import geopandas as gpd
import numpy as np


logger = logging.getLogger(__name__)


VECTOR_LINE_OPERATORS = {
    "sum": lambda geoms: sum(geom.length for geom in geoms),
    "max": lambda geoms: max(geom.length for geom in geoms),
    "min": lambda geoms: min(geom.length for geom in geoms),
    "mean": lambda geoms: np.mean([geom.length for geom in geoms]),
    "median": lambda geoms: np.median([geom.length for geom in geoms]),
    "count": lambda geoms: len(geoms),
}



def vector_line_zonal_stats(df_polygon,
                            df_line,
                            operator: str,
                            field_name: str):
    """
    Compute zonal statistics for a polygon layer from a given line layer.

    :param df_polygon: Target polygon layer dataframe which zonal statistics will be added
    :param df_line: Line layer dataframe which is used to compute zonal statistics
    :param operator: Zonal statistics operator such as sum, min, max, mean, median, count
    :param field_name: Statistics field name to be added to target polygon layer
    :return: Output dataframe with zonal statistics
    """
    # Validate operator
    selected_operator = VECTOR_LINE_OPERATORS.get(operator)
    assert selected_operator is not None, f"Operator '{operator}' is not supported."

    # Ensure both datasets have spatial indices for efficient spatial join
    if df_polygon.geometry.name != 'geometry':
        df_polygon = df_polygon.set_geometry('geometry')

    if df_line.geometry.name != 'geometry':
        df_line = df_line.set_geometry('geometry')

    # Spatial index for faster joins
    df_polygon.sindex
    df_line.sindex

    # Before performing the spatial join, ensure there is no column named 'index_right' in both dataframes
    if 'index_right' in df_polygon.columns:
        df_polygon = df_polygon.rename(columns={'index_right': 'polygon_index'})

    if 'index_right' in df_line.columns:
        df_line = df_line.rename(columns={'index_right': 'line_index'})

    # Perform the spatial join with rsuffix to avoid column name conflicts
    intersects = gpd.sjoin(df_line, df_polygon, how="inner", predicate="intersects", rsuffix='_right')

    # Rename 'index_right' to avoid conflict with existing columns
    intersects = intersects.rename(columns={'index_right': 'polygon_index'})

    # Debugging: Check if 'polygon_index' is 1-dimensional and clean it up if necessary
    if not intersects['polygon_index'].dtype.kind in 'iufc':  # Check if not numeric or integer-like
        print("Warning: 'polygon_index' is not numeric. Converting to a simple 1-dimensional index.")
        intersects['polygon_index'] = intersects['polygon_index'].astype('category')  # Convert to category or numeric

    # Check if there are missing values or NaNs
    if intersects['polygon_index'].isnull().any():
        print("Warning: There are missing values in 'polygon_index'. Replacing with 0.")
        intersects['polygon_index'] = intersects['polygon_index'].fillna(0)

    # Ensure 'polygon_index' is 1-dimensional
    intersects = intersects.reset_index(drop=True)

    # Calculate zonal statistics by applying the selected operator on 'geometry'
    line_stats = intersects.groupby("polygon_index")["geometry"].apply(selected_operator)

    # Merge back to the polygons and assign the zonal stats
    df_polygon[field_name] = df_polygon.index.map(line_stats).fillna(0)

    return df_polygon

