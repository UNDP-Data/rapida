import asyncio
import tempfile
import httpx
import logging
import os

import pycountry
from osgeo import gdal, ogr
from rich.progress import Progress
from shapely.geometry import shape, box
import geojson

from rapida.util.http_get_json import http_get_json

gdal.UseExceptions()
CGAZ_GEOBOUNDARIES_ROOT = "https://www.geoboundaries.org/api/current/gbHumanitarian"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def countries_for_bbox(bounding_box=None):
    str_bbox = map(str, bounding_box)
    url = f'https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/World_Countries_(Generalized)/FeatureServer/0/query?where=1%3D1&outFields=*&geometry={",".join(str_bbox)}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&returnGeometry=false&outSR=4326&f=json'
    try:
        timeout = httpx.Timeout(connect=10, read=1800, write=1800, pool=1000)
        data = http_get_json(url=url, timeout=timeout)
        countries = [pycountry.countries.get(alpha_2=country['attributes']['ISO']).alpha_3 for country in data['features']]
        return tuple(countries)
    except Exception as e:
        logger.error(f'Failed to fetch countries that intersect bbox {bounding_box}. {e}')
        raise

async def download_geojson(directory, client, url, country, level, progress, task_id):
    try:
        progress.console.log(f"[yellow]Fetching metadata for {country} ADM{level} from {url}")
        response = await client.get(url)
        response.raise_for_status()
        progress.update(task_id, advance=10)

        data = response.json()
        geojson_url = data.get('gjDownloadURL')
        if not geojson_url:
            progress.console.log(f"[red]No GeoJSON URL found for {country} ADM{level}")
            return None
        progress.console.log(f"[yellow]Downloading GeoJSON from {geojson_url}")
        progress.update(task_id, advance=10)

        geojson_path = os.path.join(directory, f"{country}_ADM{level}.geojson")
        async with client.stream("GET", geojson_url, follow_redirects=True) as stream_resp:
            stream_resp.raise_for_status()
            with open(geojson_path, 'wb') as f:
                async for chunk in stream_resp.aiter_bytes():
                    f.write(chunk)
        progress.update(task_id, completed=100)
        progress.console.log(f"[green]Saved {country} ADM{level} to {geojson_path}")
        return geojson_path
    except Exception as exc:
        progress.console.log(f"[red]Error fetching {country} ADM{level}: {exc}")
        return None

def merge_geojson_files(input_paths, output_path, admin_level=1, progress=None):
    drv = ogr.GetDriverByName("GeoJSON")
    if os.path.exists(output_path):
        drv.DeleteDataSource(output_path)

    base_ds = drv.Open(input_paths[0])
    if not base_ds:
        raise RuntimeError(f"Could not open base dataset: {input_paths[0]}")
    base_layer = base_ds.GetLayer()

    out_ds = drv.CreateDataSource(output_path)
    out_layer = out_ds.CreateLayer(f"admin{admin_level}", srs=base_layer.GetSpatialRef(), geom_type=base_layer.GetGeomType())

    layer_defn = base_layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        out_layer.CreateField(layer_defn.GetFieldDefn(i))

    out_defn = out_layer.GetLayerDefn()

    if progress:
        task_id = progress.add_task("Merging GeoJSON files", total=len(input_paths))

    for path in input_paths:
        ds = drv.Open(path)
        if not ds:
            if progress:
                progress.console.log(f"[red]Could not open {path}, skipping")
            continue
        layer = ds.GetLayer()
        for feature in layer:
            out_feat = ogr.Feature(out_defn)
            out_feat.SetGeometry(feature.GetGeometryRef().Clone())
            for i in range(out_defn.GetFieldCount()):
                out_feat.SetField(out_defn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))
            out_layer.CreateFeature(out_feat)
        ds = None
        if progress:
            progress.update(task_id, advance=1)

    out_ds = None
    return geojson.load(open(output_path))

async def fetch_admin(bbox=None, admin_level=None, clip=False):
    intersecting_countries = countries_for_bbox(bounding_box=bbox)
    if not intersecting_countries:
        logger.info(f'The supplied bounding box {bbox} contains no countries')
        return None

    admin_levels = dict(zip(intersecting_countries, [admin_level] * len(intersecting_countries)))
    logger.info(f'Fetching admin boundaries for: {admin_levels}')
    geojson_paths = []

    with tempfile.TemporaryDirectory(delete=False) as tmpdirname:
        logger.info(f'Created temporary directory {tmpdirname}')
        async with httpx.AsyncClient() as client:
            with Progress() as progress:
                tasks = []
                for country, level in admin_levels.items():
                    url = f"{CGAZ_GEOBOUNDARIES_ROOT}/{country}/ADM{level}/"
                    task_id = progress.add_task(f"[cyan]Downloading {country} ADM{level}", total=100)
                    task = download_geojson(tmpdirname, client, url, country, level, progress, task_id)
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, str) and os.path.exists(result):
                        geojson_paths.append(result)
                    elif isinstance(result, Exception):
                        progress.console.log(f"[red]Error during GeoJSON fetch: {result}")

                if not geojson_paths:
                    progress.console.log("[red]No GeoJSON files were downloaded.")
                    return None

                merged = merge_geojson_files(
                    geojson_paths,
                    admin_level=admin_level,
                    output_path=f'/tmp/admin{admin_level}.geojson',
                    progress=progress
                )

                if clip:
                    west, south, east, north = bbox
                    bbox_polygon = box(west, south, east, north)
                    for feature in merged['features']:
                        geom = shape(feature['geometry'])
                        geom = geom.intersection(bbox_polygon)
                        feature['geometry'] = geom.__geo_interface__

                for feature in merged['features']:
                    props = feature.get('properties', {})
                    feature['properties'] = {
                        'undp_admin_level': admin_level,
                        'h3id': props.get('shapeID', None),
                        'iso3': props.get('shapeGroup', None),
                        'admin1_name': props.get('shapeName', None),
                    }

                # cleanup
                for path in geojson_paths:
                    if os.path.exists(path):
                        os.remove(path)

                output_path = f'/tmp/admin{admin_level}.geojson'
                if os.path.exists(output_path):
                    os.remove(output_path)

                progress.console.log(f"[green]Fetched admin boundaries for {len(admin_levels)} countries.")
                return merged


if __name__ == "__main__":
    bbox = [22.126465, 2.306506, 32.277832, 8.863362]
    admin_level = 2
    result = asyncio.run(fetch_admin(bbox=bbox, admin_level=admin_level, clip=False))
    print(result)
