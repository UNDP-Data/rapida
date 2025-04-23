from collections import defaultdict
from concurrent.futures import as_completed
import logging
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
import geopandas as gpd
from rich.progress import Progress


def chunk_gdf(gdf, chunk_size):
    for i in range(0, len(gdf), chunk_size):
        yield gdf.iloc[i:i + chunk_size]


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

    print(chunk_areas)

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


def run_overlay(polygons_df=None, overlay_df=None, default_chunk_size=2, **kwargs):
    """
    Run overlay function
    :param polygons_df: The polygon geometries dataframe
    :param overlay_df: The overlay dataframe
    :param default_chunk_size: The default chunk size
    :param kwargs:
    :return:
    """
    progress = kwargs.get('progress')
    max_workers = 4
    final_results = gpd.GeoDataFrame()
    overlay_df_sindex = overlay_df.sindex
    chunks = list(dynamic_chunk_df(polygons_df, chunk_size=default_chunk_size))

    if progress:
        overlay_task = progress.add_task(description=f'[red]Running overlay for {len(chunks)} chunks', total=len(chunks))
    else:
        progress = Progress()
        overlay_task = progress.add_task(description=f'[green]Running overlay for {len(chunks)} chunks', total=len(chunks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_chunk, chunk, overlay_df, overlay_df_sindex, polygons_df.crs)
            for chunk in chunks
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
                if not result.empty:
                    result_gdf = gpd.GeoDataFrame(result, geometry='geometry', crs=polygons_df.crs)
                    final_results = pd.concat([final_results, result_gdf], ignore_index=True)
                    progress.update(overlay_task, advance=1)
            except Exception as e:
                logging.error(f"Error processing chunk: {e}")
    if len(final_results) > 0:
        if progress and overlay_task:
            progress.remove_task(overlay_task)
        logging.info("run_overlay complete")
        return final_results
    else:
        logging.warning("No intersecting rows from overlay_df found for any polygons.")

