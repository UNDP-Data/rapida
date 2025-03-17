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
    Compute zonal statistics for polygon layer from a given line layer

    :param df_polygon: Target polygon layer dataframe which zonal statistics is added
    :param df_line: Line layer dataframe which is used to compute zonal statistics
    :param operator: zonal statistics operator such as sum, min, max, mean, median, count.
    :param field_name: statistics field name to be added to target polygon layer
    :return: output dataframe with zonal statistics
    """
    df_output = df_polygon.copy()

    selected_operator = VECTOR_LINE_OPERATORS.get(operator)

    # if operator == 'divide':
    #     df_output['area'] = df_output.geometry.area
    #     selected_operator = lambda geom: geom.length / df_output['area']

    assert selected_operator is not None, f"Operator '{operator}' is not supported."

    df_line_cloned = df_line.copy()
    intersects = gpd.sjoin(df_line_cloned, df_output, predicate="intersects", how="inner")
    line_stats = intersects.groupby("index_right")["geometry"].apply(selected_operator)
    df_output[field_name] = df_output.index.map(line_stats).fillna(0)

    return df_output