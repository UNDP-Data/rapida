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

def chunk_gdf_by_size(gdf, chunk_size):
    """
    Split GeoDataFrame into chunks of specified size.
    :param gdf: GeoDataFrame to be chunked
    :param chunk_size: Number of rows per chunk
    :yield: GeoDataFrame chunks
    """
    for start in range(0, len(gdf), chunk_size):
        yield gdf.iloc[start:start + chunk_size]



def run_overlay(
    polygons_data_path=None,
    polygons_layer_name=None,
    input_data_path=None,
    input_layer_name=None,
    default_chunk_size=10,
    input_chunk_size=50000,
    **kwargs
):
    logging.info("Starting overlay process")

    if polygons_data_path is None or polygons_layer_name is None:
        raise ValueError("polygons_data_path and polygons_layer_name must be provided")
    if input_data_path is None or input_layer_name is None:
        raise ValueError("input_data_path and input_layer_name must be provided")

    sample_input_chunk = next(gdf_chunks(input_data_path, chunk_size=1, layer_name=input_layer_name))
    input_columns = sample_input_chunk.columns.tolist()

    final_results = gpd.GeoDataFrame()
    max_workers = kwargs.get('max_workers', 4)
    progress = kwargs.get('progress')

    input_data_chunks = []
    input_sindexes = []
    if progress:
        spatial_indexing_task = progress.add_task("[red]Building spatial index...", total=input_chunk_size)
    else:
        progress = Progress()
        spatial_indexing_task = progress.add_task(description=f'[green]Building spatial index....', total=input_chunk_size)
    for input_chunk_index, input_data_chunk in enumerate(
        gdf_chunks(input_data_path, chunk_size=input_chunk_size, layer_name=input_layer_name)
    ):
        input_sindex = input_data_chunk.sindex
        input_data_chunks.append(input_data_chunk)
        input_sindexes.append(input_sindex)
        if progress and spatial_indexing_task is not None:
            progress.update(spatial_indexing_task, advance=1)
    progress.remove_task(spatial_indexing_task)
    logging.info('Spatial index built')

    polygon_chunks = list(gdf_chunks(polygons_data_path, chunk_size=default_chunk_size, layer_name=polygons_layer_name))
    if progress:
        overlay_task = progress.add_task("[red]Running overlay...", total=len(polygon_chunks))
    else:
        progress = Progress()
        overlay_task = progress.add_task(description=f'[green]Running overlay....', total=len(polygon_chunks))

    for chunk_index, polygon_chunk in enumerate(polygon_chunks):
        chunks_rows = list(dynamic_chunk_df(polygon_chunk, chunk_size=10))
        for input_chunk_index, input_data_chunk in enumerate(input_data_chunks):
            input_chunk_sindex = input_sindexes[input_chunk_index]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for row_index, chunk_row in enumerate(chunks_rows):
                    futures.append(
                        executor.submit(
                            process_chunk,
                            chunk_row,
                            input_data_chunk,
                            input_chunk_sindex,
                            polygon_chunk.crs
                        )
                    )

                for future in as_completed(futures):
                    result = future.result()
                    if not result.empty:
                        result_gdf = gpd.GeoDataFrame(result, geometry='geometry', crs=polygon_chunk.crs)
                        final_results = pd.concat([final_results, result_gdf], ignore_index=True)

        # Drop unnecessary columns
        poly_cols = gdf_columns(file_path=polygons_data_path, layer_name=polygons_layer_name)
        cols_to_drop = set(poly_cols).difference({'h3id'}).difference(input_columns)
        final_results.drop(columns=list(cols_to_drop), inplace=True, errors='ignore')

        if progress and overlay_task is not None:
            progress.update(overlay_task, advance=1)

    logging.info('Overlay process completed')
    progress.remove_task(overlay_task)
    return final_results


