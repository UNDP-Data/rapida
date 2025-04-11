from concurrent.futures import ProcessPoolExecutor
import logging
import pandas as pd
import geopandas as gpd
from rich.progress import Progress


def chunk_gdf(gdf, chunk_size):
    for i in range(0, len(gdf), chunk_size):
        yield gdf.iloc[i:i + chunk_size]

def process_chunk(chunk_rows, roads_layer, roads_sindex, crs):
    results = []

    for idx, row in chunk_rows.iterrows():
        single_row_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=crs)
        poly_geom = single_row_gdf.geometry.iloc[0]

        possible_matches_index = list(roads_sindex.intersection(poly_geom.bounds))
        possible_matches = roads_layer.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(poly_geom)].copy()
        precise_matches.loc[:, "geometry"] = precise_matches.geometry.intersection(poly_geom)

        for col in single_row_gdf.columns:
            if col != "geometry":
                precise_matches.loc[:, col] = single_row_gdf.iloc[0][col]

        if not precise_matches.empty:
            results.append(precise_matches)

    if results:
        return pd.concat(results, ignore_index=True)

    return pd.DataFrame(columns=roads_layer.columns)

def run_overlay(polygons_df=None, overlay_df=None, chunk_size=100, **kwargs):
    progress = kwargs.get('progress')
    # chunk_size = 2
    max_workers = 4
    final_results = gpd.GeoDataFrame()
    roads_sindex = overlay_df.sindex
    chunks = list(chunk_gdf(polygons_df, chunk_size))
    if progress:
        overlay_task = progress.add_task(description=f'[red]Running overlay for {len(chunks)} chunks')
    else:
        progress = Progress()
        overlay_task = progress.add_task(description=f'[green]Running overlay for {len(chunks)} chunks')
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_chunk, chunk, overlay_df, roads_sindex, polygons_df.crs)
            for chunk in chunks
        ]
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if not result.empty:
                    result_gdf = gpd.GeoDataFrame(result, geometry='geometry', crs=polygons_df.crs)
                    final_results = pd.concat([final_results, result_gdf], ignore_index=True)
                    progress.update(overlay_task, advance=i)
            except Exception as e:
                logging.error(f"Error processing chunk {i + 1}: {e}")
    if len(final_results) > 0:
        return final_results
    else:
        logging.warning("No intersecting roads found for any polygons.")
    logging.info("run_overlay complete")


