from urllib.parse import urlparse
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import os
import httpx
from rapida.ntl.nasa.io import download_tile
from rapida.connectivity.runcli import run_cli
from pathlib import Path
import asyncio
import logging
from rich.progress import Progress
import json
from shapely.geometry import shape, mapping
from osgeo import gdal, ogr, osr
from shapely.wkb import loads as load_wkb
from shapely.ops import orient, transform
import numpy as np
from pyproj import Transformer

gdal.UseExceptions()
logger = logging.getLogger(__name__)

project_to_meters = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
project_to_degrees = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform


async def fetch_geofabrik_index(client: httpx.AsyncClient) -> dict:
    """Fetch the master spatial metadata tree from Geofabrik."""
    response = await client.get("https://download.geofabrik.de/index-v1.json")
    response.raise_for_status()
    return response.json()


async def prepare_osm_pbf_old(bbox: tuple[float, float, float, float], dst_dir: str = "/tmp", progress: Progress = None,
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
            fr_url = url.replace("https://download.geofabrik.de", "http://download.openstreetmap.fr/extracts")
            purl = urlparse(fr_url)
            file_name = os.path.basename(purl.path)
            filepath = dest_path / file_name

            tasks.append(asyncio.Task(
                download_tile(client, fr_url, filepath, semaphore, progress=progress), name=file_name
            ))

        if progress and pbf_urls:
            progress_task = progress.add_task(description=f'[red]Downloading {len(tasks)} pbf(s)...', total=len(tasks))

        for task in asyncio.as_completed(tasks, timeout=1800 * len(tasks)):
            try:
                downloaded_file = await task
                if progress and progress_task is not None and pbf_urls:
                    progress.update(progress_task, description=f'[green]🡇 Downloaded {downloaded_file.name}', advance=1)
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


async def prepare_osm_pbf(bbox: tuple[float, float, float, float], dst_dir: str = "/tmp", progress: Progress = None,
                          max_concurrency: int = 5, use_geofabrik: bool = True) -> str:
    """
    Orchestrates the entire top-down OSM extraction pipeline, supporting both Geofabrik and Movisda.
    """
    minx, miny, maxx, maxy = bbox
    bbox_geom = box(minx, miny, maxx, maxy)
    bbox_str = f"{minx},{miny},{maxx},{maxy}"

    dest_path = Path(dst_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    final_output_pbf = str(dest_path / "local_routing.osm.pbf")
    downloaded_files = []

    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
        if use_geofabrik:
            response = await client.get("https://download.geofabrik.de/index-v1.json")
            response.raise_for_status()
            features = [
                f for f in response.json().get('features', [])
                if f['properties'].get('iso3166-1:alpha2') or f['properties'].get('iso3166-2')
            ]
        else:
            response = await client.get("https://osm.download.movisda.io/admin/Admin-latest.geojson")
            response.raise_for_status()
            features = [
                f for f in response.json().get('features', [])
                if str(f['properties'].get('admin_level')) == "2"
            ]

        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        intersecting = gdf[gdf.intersects(bbox_geom)]

        if intersecting.empty:
            source = "Geofabrik" if use_geofabrik else "Movisda"
            raise ValueError(f"No valid {source} footprints cover the bbox: {bbox}")

        pbf_urls = []
        if use_geofabrik:
            for url in intersecting['urls'].apply(lambda x: x.get('pbf')).dropna():
                # Routing to the French mirror as originally specified
                try:
                    r = await client.head(url=url, follow_redirects=True)
                    r.raise_for_status()
                    pbf_urls.append(url)
                except httpx.HTTPError:
                    fr_url = url.replace("https://download.geofabrik.de", "http://download.openstreetmap.fr/extracts")
                    pbf_urls.append(fr_url)

        else:
            for _, row in intersecting.iterrows():
                name = row.get('name')
                prefix = row.get('prefix')
                timestamp = row.get('timestamp')

                actual_prefix = prefix if pd.notna(prefix) and prefix else f"{name}-"
                pbf_urls.append(f"https://osm.download.movisda.io/admin/{actual_prefix}{timestamp}.osm.pbf")

        tasks = []
        semaphore = asyncio.Semaphore(max_concurrency)

        for url in pbf_urls:
            file_name = os.path.basename(urlparse(url).path)
            filepath = dest_path / file_name

            tasks.append(asyncio.Task(
                download_tile(client, url, filepath, semaphore, progress=progress), name=file_name
            ))

        if progress and pbf_urls:
            progress_task = progress.add_task(description=f'[red]Downloading {len(tasks)} pbf(s)...', total=len(tasks))

        for task in asyncio.as_completed(tasks, timeout=1800 * len(tasks)):
            try:
                downloaded_file = await task
                if progress and progress_task is not None and pbf_urls:
                    progress.update(progress_task, description=f'[green]🡇 Downloaded {downloaded_file.name}', advance=1)
                downloaded_files.append(str(downloaded_file))
            except Exception as e:
                logger.error(e)
                raise
            except asyncio.CancelledError:
                for atask in tasks:
                    if not atask.done():
                        atask.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
    if progress and pbf_urls:
        progress.remove_task(progress_task)
        # ---------------------------------------------------------
        # THE FIX: Extract FIRST, Filter Empty, Merge SECOND
        # ---------------------------------------------------------
        extracted_chunks = []

        # 1. Extract the bounding box from each downloaded country file individually
        for i, dl_file in enumerate(downloaded_files):
            ext_file = os.path.join(dst_dir, f"ext_chunk_{i}.osm.pbf")

            # Explicitly use complete_ways to ensure boundary routing integrity
            run_cli([
                "osmium", "extract", "--overwrite", "-s", "complete_ways",
                "-b", bbox_str, dl_file, "-o", ext_file
            ])

            # Guard: Check if the extracted file actually contains data.
            # An empty osmium PBF header is usually ~105-150 bytes.
            if os.path.exists(ext_file) and os.path.getsize(ext_file) > 250:
                extracted_chunks.append(ext_file)
            else:
                logger.info(f"Chunk for {dl_file} is empty (no OSM data in this bbox). Discarding.")
                if os.path.exists(ext_file):
                    os.remove(ext_file)

        # 2. Merge the valid chunks
        if len(extracted_chunks) > 1:
            run_cli(["osmium", "merge", "--overwrite"] + extracted_chunks + ["-o", final_output_pbf])

            # Clean up the intermediate chunks
            for path in extracted_chunks:
                os.remove(path)

        elif len(extracted_chunks) == 1:
            # If there was only one valid chunk, just rename it to the final output target
            os.rename(extracted_chunks[0], final_output_pbf)

        else:
            raise ValueError(f"No OSM data found in the provided bbox: {bbox}")

        # Optional: Clean up the massive downloaded country files to save disk space
        # for dl_file in downloaded_files:
        #     if os.path.exists(dl_file):
        #         os.remove(dl_file)

        return final_output_pbf


async def extract_health_sites(pbf_path: str, dst_dir: str, progress=None) -> str:
    """
    Extracts multi-category health infrastructure by filtering and exporting
    via Osmium, processing polygon centroids cleanly via Shapely.
    """
    dst_path = Path(dst_dir)
    filtered_pbf = dst_path / "health_sites.osm.pbf"
    raw_geojson = dst_path / "raw_health_sites.geojson"
    final_geojson = dst_path / "health_sites.geojson"

    tags_to_keep = [
        "nwr/amenity=hospital,clinic,doctors,pharmacy,dentist",
        "nwr/healthcare"
    ]

    if progress:
        progress.console.print("[cyan]Filtering OSM graph and exporting to GeoJSON via Osmium...[/cyan]")

    # Step 1: Filter the PBF down to the required health elements
    run_cli(["osmium", "tags-filter", pbf_path] + tags_to_keep + ["-o", str(filtered_pbf), "--overwrite"])

    # Step 2: Export directly to GeoJSON (handles all geometries and nested tags natively)
    run_cli(["osmium", "export", "--overwrite", str(filtered_pbf), "-o", str(raw_geojson)])

    # Step 3: Compute centroids and flatten properties in a background thread
    def process_geometries():
        with open(raw_geojson, "r") as f:
            data = json.load(f)

        processed_features = []
        for i, feature in enumerate(data.get("features", []), start=1):
            # Parse geometry using shapely
            geom = shape(feature["geometry"])

            # Force everything (Polygons, MultiPolygons, Lines) to a Point centroid
            if geom.geom_type != "Point":
                geom = geom.centroid

            # Osmium export nests all attributes under a clean 'tags' dictionary
            tags = feature["properties"].get("tags", {})

            processed_features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "osm_id": feature["properties"].get("id", i),
                    "osm_type": feature["properties"].get("type"),
                    "name": tags.get("name"),
                    "amenity": tags.get("amenity"),
                    "healthcare": tags.get("healthcare")
                }
            })

        data["features"] = processed_features

        with open(final_geojson, "w") as f:
            json.dump(data, f)

    if progress:
        progress.console.print("[cyan]Computing centroids and matching schemas...[/cyan]")

    await asyncio.to_thread(process_geometries)

    # Clean up intermediates safely
    for path in [filtered_pbf, raw_geojson]:
        if path.exists():
            path.unlink()

    if progress:
        progress.console.print(f"[bold green]✓ Health sites successfully extracted to: {final_geojson}[/bold green]")

    return str(final_geojson)


def extract_origins_from_geojson(geojson_path: str) -> list[tuple[float, float]]:
    """
    Extracts a list of (longitude, latitude) tuples from a GeoJSON FeatureCollection.
    """
    with open(geojson_path, "r") as f:
        data = json.load(f)

    origins = []
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        #if feature['properties']['osm_id'] != 80:continue

        # Ensure we are only grabbing valid Points
        if geom.get("type") == "Point":
            coords = geom.get("coordinates")
            if coords and len(coords) >= 2:
                # Append as (lon, lat)
                origins.append((float(coords[0]), float(coords[1])))

    return origins


def extract_origins(sites_dataset: str=None, src_layer: str = None) -> list[tuple[float, float]]:
    """
    Extracts a list of (longitude, latitude) tuples from a spatial file using OGR.
    Handles reprojection to WGS84 (EPSG:4326) if the source is not in lat/lon.
    """
    # Open the dataset
    with gdal.OpenEx(sites_dataset, gdal.OF_VECTOR) as src_ds:

        try:
            layer = src_ds.GetLayer(int(src_layer))
        except ValueError:
            layer = src_ds.GetLayerByName(str(src_layer))

        if layer is None:
            raise ValueError(f"Layer '{src_layer}' could not be found in the dataset {sites_dataset}.")



        # Set up coordinate transformation to WGS84 (EPSG:4326)
        source_srs = layer.GetSpatialRef()
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)

        # OGR 3+ strict axis mapping strategy (ensures Longitude/Latitude order)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        transform = None
        if source_srs and not source_srs.IsSame(target_srs):
            transform = osr.CoordinateTransformation(source_srs, target_srs)

        origins = []
        layer.ResetReading()
        # Iterate through features
        for feature in layer:
            # If you still need the filter: if feature.GetField("osm_id") != 80: continue

            geom = feature.GetGeometryRef()
            if geom is not None:
                # Clone geometry to avoid modifying original layer data during transform
                geom_clone = geom.Clone()

                # Reproject if necessary
                if transform:
                    geom_clone.Transform(transform)

                # Helper function to recursively extract points from nested collections
                def extract_points(g):
                    name = g.GetGeometryName()
                    if name == "POINT":
                        origins.append((g.GetX(), g.GetY()))
                    elif name in ("MULTIPOINT", "GEOMETRYCOLLECTION"):
                        for i in range(g.GetGeometryCount()):
                            sub_geom = g.GetGeometryRef(i)
                            if sub_geom is not None:
                                extract_points(sub_geom)

                extract_points(geom_clone)

        return origins


def read_barriers_grid(src_path: str, src_layer: str = None, barriers_buffer:float=None) -> list:
    """Reads a vector source and cuts features into micro-tiles to stay under Valhalla's limit."""
    if not src_path:
        return []

    if src_layer is None:
        src_layer = "0"

    exclude_polygons = []
    GRID_SIZE = 0.01  # ~1.1 km step size in degrees. Guarantees perimeters stay tiny.

    with (gdal.OpenEx(str(src_path), gdal.OF_VECTOR | gdal.OF_READONLY) as src_ds):
        if src_ds is None:
            raise FileNotFoundError(f"GDAL could not open data source: {src_path}")

        try:
            lyr = src_ds.GetLayer(int(src_layer))
        except ValueError:
            lyr = src_ds.GetLayerByName(str(src_layer))

        if lyr is None:
            raise ValueError(f"Layer '{src_layer}' could not be found in the dataset.")

        lyr.ResetReading()
        for feat in lyr:
            geom_ogr = feat.GetGeometryRef()
            if geom_ogr is None or geom_ogr.IsEmpty():
                continue


            wkb_output = geom_ogr.ExportToWkb()
            if isinstance(wkb_output, int) or wkb_output is None:
                continue

            raw_geom = load_wkb(bytes(wkb_output))
            # buffer lines

            if 'line' in raw_geom.geom_type.lower():
                assert barriers_buffer is not None, f'Invalid barriers_buffer={barriers_buffer}'
                raw_geom = transform(project_to_degrees, transform(project_to_meters, raw_geom).buffer(distance=barriers_buffer))

            # Add this block to handle giant natural polygons??? in v
            # elif 'polygon' in raw_geom.geom_type.lower():
            #     raw_geom = transform(
            #         project_to_degrees,
            #         # Simplify by 5-10 meters to drastically reduce coordinate count
            #         transform(project_to_meters, raw_geom).simplify(5.0)
            #     )

            geoms_to_process = [raw_geom] if raw_geom.geom_type == "Polygon" else list(raw_geom.geoms)


            for geom in geoms_to_process:
                minx, miny, maxx, maxy = geom.bounds

                # Create a uniform grid over the polygon's bounding box extent
                x_coords = np.arange(minx, maxx + GRID_SIZE, GRID_SIZE)
                y_coords = np.arange(miny, maxy + GRID_SIZE, GRID_SIZE)

                for i in range(len(x_coords) - 1):
                    for j in range(len(y_coords) - 1):
                        grid_cell = box(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])

                        # Only intersection geometry inside this 1km micro-cell
                        if geom.intersects(grid_cell):
                            intersected_part = geom.intersection(grid_cell)

                            if not intersected_part.is_empty and intersected_part.geom_type in (
                            "Polygon", "MultiPolygon"):
                                sub_polys = [intersected_part] if intersected_part.geom_type == "Polygon" else list(
                                    intersected_part.geoms)

                                for sub_poly in sub_polys:
                                    # Ensure ring winding order is correct
                                    ccw_poly = orient(sub_poly, sign=1.0)
                                    ring_coords = [[float(pt[0]), float(pt[1])] for pt in ccw_poly.exterior.coords]

                                    # Nest inside an extra array for the Isochrone engine
                                    exclude_polygons.append(ring_coords)

        lyr = None

    return exclude_polygons


def read_barriers(src_path: str, src_layer: str = None, barriers_buffer: float = None) -> list:
    if not src_path:
        return []

    src_layer = str(src_layer) if src_layer is not None else "0"
    exclude_polygons = []

    with gdal.OpenEx(str(src_path), gdal.OF_VECTOR | gdal.OF_READONLY) as src_ds:
        if src_ds is None:
            raise FileNotFoundError(f"GDAL could not open data source: {src_path}")

        try:
            lyr = src_ds.GetLayer(int(src_layer))
        except ValueError:
            lyr = src_ds.GetLayerByName(src_layer)

        if lyr is None:
            raise ValueError(f"Layer '{src_layer}' could not be found.")

        lyr.ResetReading()
        for feat in lyr:
            geom_ogr = feat.GetGeometryRef()
            if geom_ogr is None or geom_ogr.IsEmpty():
                continue

            wkb_output = geom_ogr.ExportToWkb()
            if wkb_output is None or isinstance(wkb_output, int):
                continue

            raw_geom = load_wkb(bytes(wkb_output))

            if 'line' in raw_geom.geom_type.lower():
                assert barriers_buffer is not None, f'Invalid barriers_buffer={barriers_buffer}'
                # Buffer and simplify to prevent exceeding Valhalla limits
                raw_geom = transform(
                    project_to_degrees,
                    transform(project_to_meters, raw_geom).buffer(distance=barriers_buffer).simplify(5.0)
                )

            geoms_to_process = [raw_geom] if raw_geom.geom_type == "Polygon" else list(getattr(raw_geom, 'geoms', []))

            for sub_poly in geoms_to_process:
                if sub_poly.geom_type not in ("Polygon", "MultiPolygon"):
                    continue

                ccw_poly = orient(sub_poly, sign=1.0)
                ring_coords = [[float(pt[0]), float(pt[1])] for pt in ccw_poly.exterior.coords]
                exclude_polygons.append(ring_coords)

    return exclude_polygons