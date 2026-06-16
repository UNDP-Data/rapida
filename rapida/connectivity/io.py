from urllib.parse import urlparse
import geopandas as gpd
from shapely.geometry import box
import os
import httpx
from rapida.ntl.nasa.io import download_tile
from rapida.connectivity.runcli import run_cli
from pathlib import Path
import asyncio
import logging
from rich.progress import Progress

logger = logging.getLogger(__name__)




async def fetch_geofabrik_index(client: httpx.AsyncClient) -> dict:
    """Fetch the master spatial metadata tree from Geofabrik."""
    response = await client.get("https://download.geofabrik.de/index-v1.json")
    response.raise_for_status()
    return response.json()


async def prepare_osm_pbf(bbox: tuple[float, float, float, float], dst_dir: str = "/tmp", progress: Progress = None,
                          max_concurrency: int = 5) -> str:
    """
    Orchestrates the entire top-down OSM extraction pipeline.
    """
    # 1. FIX: Handle the tuple correctly
    minx, miny, maxx, maxy = bbox
    bbox_geom = box(minx, miny, maxx, maxy)
    bbox_str = f"{minx},{miny},{maxx},{maxy}"  # For the C++ CLI

    dest_path = Path(dst_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    final_output_pbf = str(dest_path / "local_routing.osm.pbf")
    downloaded_files = []

    # Establish shared connection pool context
    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
        index_data = await fetch_geofabrik_index(client)

        # Filter features to bypass massive macro-continent entries
        features = [
            f for f in index_data['features']
            if f['properties'].get('iso3166-1:alpha2') or f['properties'].get('iso3166-2')
        ]

        # Perform quick spatial overlay check
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        intersecting = gdf[gdf.intersects(bbox_geom)]

        if intersecting.empty:
            raise ValueError(f"No valid Geofabrik country footprints cover the bbox: {bbox}")

        tasks = []
        semaphore = asyncio.Semaphore(max_concurrency)
        pbf_urls = intersecting['urls'].apply(lambda x: x['pbf']).tolist()

        for url in pbf_urls:
            file_name = os.path.basename(urlparse(url).path)
            filepath = dest_path / file_name

            tasks.append(asyncio.Task(
                download_tile(client, url, filepath, semaphore, progress=progress), name=file_name
            ))

        if progress:
            progress_task = progress.add_task(description=f'Downloading {len(tasks)} pbf(s)...', total=len(tasks))

        for task in asyncio.as_completed(tasks, timeout=1800 * len(tasks)):
            try:
                downloaded_file = await task
                if progress and progress_task is not None:
                    progress.update(progress_task, description=f'[green]🡇 {downloaded_file.name}', advance=1)
                downloaded_files.append(str(downloaded_file))
            except Exception as e:
                logger.error(e)
                raise
            except asyncio.CancelledError as ce:
                for atask in tasks:
                    if not atask.done():
                        atask.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    # 2. FIX: Typo corrected to downloaded_files
    if len(downloaded_files) > 1:
        merged_pbf = os.path.join(dst_dir, "merged_source.osm.pbf")
        run_cli(["osmium", "merge", "--overwrite"] + downloaded_files + ["-o", merged_pbf])

        for path in downloaded_files:
            os.remove(path)
        input_source = merged_pbf
    else:
        input_source = downloaded_files[0]

    # 3. FIX: Pass bbox_str into subprocess list
    run_cli([
        "osmium", "extract", "--overwrite", "-b", bbox_str, input_source, "-o", final_output_pbf
    ])

    # if os.path.exists(input_source):
    #     os.remove(input_source)

    return final_output_pbf