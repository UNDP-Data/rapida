import json
import logging
import concurrent.futures
import threading
from typing import Optional, Any, Dict, List
from calendar import monthrange
from datetime import date
from rich.progress import Progress
import pystac
import pystac_client
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point, MultiPoint, Polygon, shape, mapping
from shapely.ops import unary_union
from collections import defaultdict
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
                     cloud_cover: int = 5,
                     max_workers: int = 5,
                     progress: Progress = None,
                     ):
        """
        Search stac items for covering project geodataframe area

        :param collection_id: collection id
        :param target_year: target year
        :param target_month: target month
        :param duration: how many months to search for
        :param cloud_cover: how much minimum cloud cover rate to search for. Default is 5.
        :param target_assets: target assets
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

        all_items = {}
        lock = threading.Lock()

        merged_geom = unary_union(df_polygon['geometry'])
        intersects_geometry = mapping(merged_geom)

        query = {}
        if cloud_cover == 0:
            query["eq"] = cloud_cover
        else:
            query["lt"] = cloud_cover

        def search_single_polygon(single_geom):
            nonlocal all_items

            fc_geojson_str = gpd.GeoDataFrame(geometry=[single_geom], crs=df_polygon.crs).to_json()
            fc_geojson = json.loads(fc_geojson_str)
            first_feature = fc_geojson["features"][0]

            search_query = {"eo:cloud_cover": query}
            logger.debug(f"search query: {search_query}")
            search = self.client.search(
                collections=[collection_id],
                intersects=first_feature,
                query=search_query,
                datetime=datetime_range,
            )

            local_items = []
            for item in search.items():
                percentage = self._intersection_percent(item, intersects_geometry)
                cloud_cover = item.properties.get("eo:cloud_cover")
                logger.debug(f"{item.id}: intersects) {percentage:.2f}%; cloud cover) {cloud_cover:.2f}%")

                # only use items which covers more than 10% of the entire project area
                if percentage > 10:
                    # if item does not have all required assets, skip it.
                    if all(asset in item.assets for asset in target_assets):
                        local_items.append({
                            'item': item,
                            'percentage': percentage,
                            'cloud_cover': cloud_cover
                        })

            with lock:
                for item_data in local_items:
                    all_items[item_data['item'].id] = item_data

            if progress and search_task:
                progress.update(search_task, advance=1)

        work_geoms = self._merge_or_voronoi(df_polygon)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(search_single_polygon, geom) for geom in work_geoms]
            concurrent.futures.wait(futures)

        if progress and search_task:
            progress.remove_task(search_task)

        # Apply grid:code optimization
        optimized_items = self._optimize_by_grid_code(all_items, intersects_geometry)

        return optimized_items

    def _optimize_by_grid_code(self, all_items: Dict[str, Dict], intersects_geometry: Dict[str, Any]) -> Dict[
        str, pystac.Item]:
        """
        Optimize item selection by grid:code grouping and coverage analysis.

        :param all_items: Dictionary of all found items with their metadata
        :param intersects_geometry: Target geometry for coverage calculation
        :return: Optimized dictionary of selected items
        """
        # Group items by grid:code
        grid_groups = defaultdict(list)

        for item_id, item_data in all_items.items():
            item = item_data['item']
            grid_code = item.properties.get('grid:code')

            if grid_code is None:
                # If no grid:code, use item_id as fallback
                grid_code = item_id
                logger.warning(f"Item {item_id} has no grid:code property, using item_id as fallback")

            grid_groups[grid_code].append({
                'item': item,
                'percentage': item_data['percentage'],
                'cloud_cover': item_data['cloud_cover'],
                'datetime': item.datetime
            })

        selected_items = {}
        aoi_geom = shape(intersects_geometry)

        for grid_code, items in grid_groups.items():
            logger.debug(f"Processing grid:code {grid_code} with {len(items)} items")

            # Sort items by cloud cover (lowest first), then by datetime (newest first)
            items.sort(key=lambda x: (x['cloud_cover'], -x['datetime'].timestamp()))

            # Check for 100% coverage items first
            perfect_coverage_items = [item for item in items if item['percentage'] >= 100.0]

            if perfect_coverage_items:
                # Select the newest item with 100% coverage
                selected_item = perfect_coverage_items[0]['item']
                selected_items[selected_item.id] = selected_item
                logger.debug(f"Grid {grid_code}: Selected item {selected_item.id} with 100% coverage")
            else:
                # No 100% coverage items, need to combine multiple items
                selected_for_grid = self._select_optimal_coverage(items, aoi_geom)
                for item in selected_for_grid:
                    selected_items[item.id] = item
                logger.debug(f"Grid {grid_code}: Selected {len(selected_for_grid)} items for optimal coverage")

        return selected_items

    def _select_optimal_coverage(self, items: List[Dict], aoi_geom: Polygon, max_item_limit=10) -> List[pystac.Item]:
        """
        Select the minimum set of items to achieve maximum coverage of the AOI.
        Items are already sorted by cloud_cover (lowest first), then datetime (newest first).

        :param items: List of item dictionaries sorted by preference (cloud_cover asc, datetime desc)
        :param aoi_geom: Area of Interest geometry
        :return: List of selected items
        """
        selected_items = []
        remaining_geom = aoi_geom

        logger.debug(f"Starting optimal coverage selection with {len(items)} items")

        for item_data in items:
            item = item_data['item']
            item_geom = shape(item.geometry)

            # Check if this item adds any new coverage
            intersection = remaining_geom.intersection(item_geom)

            if not intersection.is_empty and intersection.area > 0:
                # This item provides additional coverage
                selected_items.append(item)
                logger.debug(
                    f"Selected {item.id} (cloud_cover: {item_data['cloud_cover']:.2f}%, datetime: {item.datetime})")

                # Remove the covered area from remaining geometry
                try:
                    remaining_geom = remaining_geom.difference(item_geom)

                    # If remaining geometry becomes empty or very small, we're done
                    if remaining_geom.is_empty or remaining_geom.area < (
                            aoi_geom.area * 0.001):  # Less than 0.1% remaining
                        logger.debug(f"Achieved near-complete coverage with {len(selected_items)} items")
                        break

                except Exception as e:
                    logger.warning(f"Error in geometry difference calculation: {e}")
                    # Continue with the current selection if geometry operations fail
                    break
            else:
                logger.debug(f"Skipped {item.id} - no additional coverage provided")

            # Safety check to prevent infinite loops
            if len(selected_items) >= max_item_limit:
                logger.warning(f"Reached maximum item limit ({max_item_limit}) for grid coverage optimization")
                break

        return selected_items

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

    def _intersection_percent(self, item: pystac.Item, aoi: Dict[str, Any]) -> float:
        """
        The percentage that the Item's geometry intersects the AOI. An Item that
        completely covers the AOI has a value of 100.
        """
        geom_item = shape(item.geometry)
        geom_aoi = shape(aoi)

        intersected_geom = geom_aoi.intersection(geom_item)

        intersection_percent = (intersected_geom.area * 100) / geom_aoi.area

        return intersection_percent


if __name__ == '__main__':
    setup_logger(level=logging.DEBUG)

    stac_url = "https://earth-search.aws.element84.com/v1"
    mask_file = "/data/kigali/data/kigali.gpkg"
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