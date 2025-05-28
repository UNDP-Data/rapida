import json
import logging
import concurrent.futures
import threading
from typing import Optional
from calendar import monthrange
from datetime import date
from rich.progress import Progress
import pystac_client
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point, MultiPoint, Polygon

from rapida.components.landuse.constants import SENTINEL2_ASSET_MAP
from rapida.util.setup_logger import setup_logger

logger = logging.getLogger(__name__)


class StacCollection(object):

    @property
    def client(self) -> pystac_client.Client:
        return self._client

    def __init__(self, stac_url: str, mask_file: str, mask_layer: str):
        self._client = pystac_client.Client.open(stac_url)
        self.mask_file = mask_file
        self.mask_layer = mask_layer

    def search_items(self,
                     collection_id: str,
                     target_year: int,
                     target_month: int,
                     target_assets: dict[str, str] = SENTINEL2_ASSET_MAP,
                     duration: int = 12,
                     tile_id_prop_name: str = "grid:code",
                     max_workers: int = 5,
                     progress: Progress = None,
                     ):
        """
        Search stac items for covering project geodataframe area

        :param collection_id: collection id
        :param target_year: target year
        :param target_month: target month
        :param duration: how many months to search for
        :param target_assets: target assets
        :param tile_id_prop_name: property name used for tile id. If not found, item.id is used.
        :param max_workers: maximum number of workers
        :param progress: rich progress object
        """

        df_polygon = gpd.read_file(self.mask_file, layer=self.mask_layer)
        df_polygon.to_crs(epsg=4326, inplace=True)

        datetime_range = self._create_date_range(target_year, target_month, duration=duration)
        logger.debug(f"datetime range for searching: {datetime_range}")

        search_task = None
        if progress:
            search_task = progress.add_task(f"[green]Searching STAC Items for {datetime_range}", total=len(df_polygon))

        latest_per_tile = {}
        lock = threading.Lock()

        def search_single_polygon(single_geom):
            nonlocal latest_per_tile

            fc_geojson_str = gpd.GeoDataFrame(geometry=[single_geom], crs=df_polygon.crs).to_json()
            fc_geojson = json.loads(fc_geojson_str)
            first_feature = fc_geojson["features"][0]

            search = self.client.search(
                collections=[collection_id],
                intersects=first_feature,
                query={"eo:cloud_cover": {"lt": 5}},
                datetime=datetime_range,
            )
            items = list(search.items())

            with lock:
                for item in items:
                    tile_id = item.properties.get(tile_id_prop_name)
                    if tile_id is None:
                        tile_id = item.id

                    cloud_cover = item.properties.get("eo:cloud_cover")
                    if cloud_cover is None:
                        continue

                    # if item does not have all required assets, skip it.
                    if not all(asset in item.assets for asset in target_assets):
                        continue
                    existing = latest_per_tile.get(tile_id)
                    if existing is None:
                        latest_per_tile[tile_id] = item
                    else:
                        existing_cloud = existing.properties.get("eo:cloud_cover")
                        if existing_cloud is None:
                            latest_per_tile[tile_id] = item
                        elif cloud_cover < existing_cloud:
                            # Update item if cloud cover is lower even it is older item
                            latest_per_tile[tile_id] = item
                        elif item.datetime > existing.datetime:
                            # Otherwise, newer image is used
                            latest_per_tile[tile_id] = item

            if progress and search_task:
                progress.update(search_task, advance=1)

        work_geoms = self._merge_or_voronoi(df_polygon)
        # gdf_out = gpd.GeoDataFrame(geometry=work_geoms, crs=df_polygon.crs)
        # gdf_out.to_file("voronoi_regions.geojson", driver="GeoJSON")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(search_single_polygon, geom) for geom in work_geoms]
            concurrent.futures.wait(futures)

        if progress and search_task:
            progress.remove_task(search_task)

        return latest_per_tile


    def _create_date_range(self, target_year: int, target_month: Optional[int] = None, duration: int = 6) -> str:
        """
        Generate a date range string in the format 'YYYY-MM-DD/YYYY-MM-DD'.

        The end date is determined based on the provided target year and optional target month:
        - If target_year is None or in the future → use current year.
        - If target_month is None:
            - If target_year is current year → use today as end_date.
            - If target_year is past → use December as default month.
        - If target_month is in the future (within current year) → clamp to current month.
        - end_date = today (if current year and current/future month), else last day of target_month.
        - start_date = end_date - `duration` months (manual calculation, no external lib).

        The start date is computed by subtracting `duration` months from the end date,
        accounting for varying month lengths and year boundaries.

        :param target_year: The year of interest.
        :param target_month: The month of interest (1–12), optional.
        :param duration: Number of months to go back from the end date (default is 6).
        :return: A date range string in the format 'YYYY-MM-DD/YYYY-MM-DD'.
        """
        today = date.today()
        current_year = today.year
        current_month = today.month

        def subtract_months(from_date: date, months: int) -> date:
            year = from_date.year
            month = from_date.month - months

            while month <= 0:
                month += 12
                year -= 1

            day = min(from_date.day, monthrange(year, month)[1])
            return date(year, month, day)

        # normalize year
        if target_year is None or target_year > current_year:
            target_year = current_year

        # normalize month
        if target_month is None:
            if target_year < current_year:
                target_month = 12  # Use December for past years
        else:
            if target_year == current_year and target_month > current_month:
                target_month = current_month

                # Determine end_date
        if target_year == current_year:
            if target_month is None or target_month >= current_month:
                end_date = today
            else:
                last_day = monthrange(target_year, target_month)[1]
                end_date = date(target_year, target_month, last_day)
        else:
            if target_month is None:
                end_date = date(target_year, 12, 31)
            else:
                last_day = monthrange(target_year, target_month)[1]
                end_date = date(target_year, target_month, last_day)

        start_date = subtract_months(end_date, duration)

        return f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    def _merge_or_voronoi(self, df: gpd.GeoDataFrame, scene_size=110000) -> list[Polygon]:
        """
        Merge or split input polygons into voronoi depending on their spatial extent.

        If the total area covered by the input polygons is smaller than or equal to a Sentinel-2 scene
        (approximately 110 km x 110 km), merge all polygons into a single one for STAC search.

        If the area is larger, split it using Voronoi polygons by creating multiple points inside a merged polygon.

        :param df: Input GeoDataFrame containing search polygons.
        :param scene_size:  Approximate size of a Sentinel-2 scene in meters (default is 110000).
        :return A list of merged or split polygons to be used for STAC search.
        """
        original_crs = df.crs
        # reproject to 3857 to ease to compute area in meters.
        df_3857 = df.to_crs(epsg=3857)

        merged = df_3857.geometry.union_all()
        merged = merged.simplify(tolerance=1000, preserve_topology=False)
        total_area = merged.area

        if total_area <= scene_size ** 2:
            # Area is small enough to be covered by a single STAC search
            return [gpd.GeoSeries([merged], crs=3857).to_crs(original_crs).iloc[0]]

        # Generate internal grid points spaced equally based on scene size
        minx, miny, maxx, maxy = merged.bounds
        # create double size of scene to be able to cover the whole scene area
        spacing = scene_size * 2
        nx = int(np.ceil((maxx - minx) / spacing))
        ny = int(np.ceil((maxy - miny) / spacing))

        points = []
        for i in range(nx + 1):
            for j in range(ny + 1):
                x = minx + i * spacing
                y = miny + j * spacing
                pt = Point(x, y)
                if merged.contains(pt):
                    points.append(pt)

        if len(points) < 2:
            return [gpd.GeoSeries([merged], crs=3857).to_crs(original_crs).iloc[0]]

        multipoint = MultiPoint(points)
        voronoi_geom = shapely.voronoi_polygons(multipoint, extend_to=merged, only_edges=False)

        # clip voronoi polygons by original polygon
        polygons_3857 = [
            poly.intersection(merged)
            for poly in voronoi_geom.geoms
            if poly.is_valid and not poly.is_empty
        ]

        # reproject final outputs to original projection
        return gpd.GeoSeries(polygons_3857, crs=3857).to_crs(original_crs).tolist()


if __name__ == '__main__':
    setup_logger()

    stac_url = "https://earth-search.aws.element84.com/v1"
    mask_file = "/data/kigali_small/data/kigali_small.gpkg"
    mask_layer = "polygons"
    collection_id = "sentinel-2-l1c"

    stac_collection = StacCollection(stac_url=stac_url,
                                     mask_file=mask_file,
                                     mask_layer=mask_layer)

    with Progress() as progress:
        latest_per_tile = stac_collection.search_items(
            collection_id=collection_id,
            target_year=2025,
            target_month=4,
            duration=12,
            progress=progress, )

        logger.info(latest_per_tile)