from collections import defaultdict
from concurrent.futures import as_completed
import logging
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
import geopandas as gpd
from rich.progress import Progress

from rapida.util.fiona_chunks import gdf_chunks, gdf_columns


def dynamic_chunk_df(gdf, chunk_size):
    """
    Yield chunks of the GeoDataFrame, each with approximately `chunk_size` features,
    but distributed to keep total area balanced between chunks.

    :param gdf: GeoDataFrame with polygon geometries
    :param chunk_size: Desired number of features per chunk
    :yield: GeoDataFrame chunks
    """
    if gdf.empty:
        return

    # Project to an equal-area CRS for fair area calculation
    gdf_proj = gdf.to_crs("ESRI:54009")  # Mollweide
    gdf_proj['__area__'] = gdf_proj.geometry.area

    # Sort features from largest to smallest
    sorted_gdf = gdf_proj.sort_values('__area__', ascending=False)

    # Determine number of chunks based on chunk size
    num_chunks = max(1, (len(gdf) + chunk_size - 1) // chunk_size)

    # Initialize chunks and their area totals
    chunks = defaultdict(list)
    chunk_areas = [0.0] * num_chunks

    for idx, row in sorted_gdf.iterrows():
        # Assign to the chunk with the current smallest total area
        min_chunk_idx = chunk_areas.index(min(chunk_areas))
        chunks[min_chunk_idx].append(row.name)  # store original index
        chunk_areas[min_chunk_idx] += row['__area__']

    # Yield original-CRS chunks in order
    for i in range(num_chunks):
        yield gdf.loc[chunks[i]]



def process_chunk(chunk_rows, overlay_df, overlay_df_sindex, crs):
    results = []

    for idx, row in chunk_rows.iterrows():
        single_row_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=crs)
        poly_geom = single_row_gdf.geometry.iloc[0]

        possible_matches_index = list(overlay_df_sindex.intersection(poly_geom.bounds))
        possible_matches = overlay_df.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(poly_geom)].copy()
        precise_matches.loc[:, "geometry"] = precise_matches.geometry.intersection(poly_geom)

        for col in single_row_gdf.columns:
            if col != "geometry":
                precise_matches[col] = single_row_gdf.iloc[0][col]

        if not precise_matches.empty:
            results.append(precise_matches)

    if results:
        return pd.concat(results, ignore_index=True)

    return pd.DataFrame(columns=overlay_df.columns)


def run_overlay(polygons_data_path=None, polygons_layer_name=None, input_data_path=None, input_layer_name=None, default_chunk_size=10, **kwargs):
    """
    Run overlay function
    :param polygons_data_path: The polygon source path
    :param polygons_layer_name: The polygon layer name
    :param input_data_path: The input data path
    :param input_layer_name: The input layer name
    :param default_chunk_size: The default chunk size
    :param kwargs:
    :return:
    """

    if polygons_data_path is None or polygons_layer_name is None:
        raise ValueError("polygons_data_path and polygons_layer_name must be provided")
    if input_data_path is None or input_layer_name is None:
        raise ValueError("input_data_path and input_layer_name must be provided")

    progress = kwargs.get('progress')
    max_workers = 4
    final_results = gpd.GeoDataFrame()
    input_data_df = gpd.read_file(input_data_path, layer=input_layer_name, engine="pyogrio")
    input_data_df_sindex = input_data_df.sindex

    for chunk_df in gdf_chunks(polygons_data_path, chunk_size=default_chunk_size, layer_name=polygons_layer_name):
        chunks_rows = list(dynamic_chunk_df(chunk_df, chunk_size=default_chunk_size))
        if progress:
            overlay_task = progress.add_task(description=f'[red]Running overlay....', total=len(chunks_rows))
        else:
            progress = Progress()
            overlay_task = progress.add_task(description=f'[green]Running overlay....', total=len(chunks_rows))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_chunk, chunk_row, input_data_df, input_data_df_sindex, chunk_df.crs)
                for chunk_row in chunks_rows
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if not result.empty:
                        result_gdf = gpd.GeoDataFrame(result, geometry='geometry', crs=chunk_df.crs)
                        final_results = pd.concat([final_results, result_gdf], ignore_index=True)
                        progress.update(overlay_task, advance=1)
                        if len(result_gdf) > 0:
                            if progress and overlay_task:
                                progress.remove_task(overlay_task)
                except Exception as e:
                    logging.error(f"Error processing chunk: {e}")
        poly_cols = gdf_columns(file_path=polygons_data_path, layer_name=polygons_layer_name)
        cols_to_drop = set(poly_cols).difference(['h3id']).difference(input_data_df.columns.tolist())
        final_results.drop(columns=list(cols_to_drop), inplace=True)
    return final_results