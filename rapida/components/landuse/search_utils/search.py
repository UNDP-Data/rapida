
import concurrent
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, datetime
import pystac_client
from dateutil import parser as dateparser
from  math import floor
from rapida.components.landuse.search_utils.tiles import (_cloud_from_props, _iso_to_ts,
                                                          Candidate, get_tileinfo, midpoint_from_range
                                                          )
from rapida.components.landuse.search_utils.zones import generate_mgrs_tiles, utm_bounds
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import frechet_distance
from threading import Event
from rich.progress import Progress, TaskProgressColumn,BarColumn, TimeRemainingColumn, TextColumn


import logging
logger = logging.getLogger(__name__)



def sim(a, b, distance):
    """
    Perc similarity based on hausdorff dist between 2 shapely polys
    :param a:
    :param b:
    :param distance:
    :return:
    """
    L = max(a.area, b.area) ** 0.5  # characteristic length
    return 100 * (2.718281828 ** (-distance / L))


def expand_timerange(start_date: str, end_date: str, days: int = 7) -> tuple[str, str]:
    start_dt = dateparser.parse(start_date)
    end_dt = dateparser.parse(end_date)
    today = datetime.today()

    expanded_start = start_dt - timedelta(days=days)
    expanded_end = end_dt + timedelta(days=days)

    if expanded_end > today:
        expanded_end = today

    return expanded_start.date().isoformat(), expanded_end.date().isoformat()


def search( client=None, collection="sentinel-2-l1c",
            start_date=None,end_date=None, mgrs_id=None,max_cloud_cover=10,
            stop:Event = None, progress:Progress=None, prune=False
            ):
    """

    :param client:
    :param collection:
    :param start_date:
    :param end_date:
    :param mgrs_id:
    :return:
    """
    mid = midpoint_from_range(range_str=f"{start_date}/{end_date}" if start_date and end_date else None)
    prev_coverage = 0
    mgrs_poly, crs = utm_bounds(mgrs_id)

    search_progress = None

    while True:
        if progress is not None:
            if search_progress is None:
                search_progress = progress.add_task(
                    description=f'[red]Searching for Sentinel2 data between {start_date} and {end_date} and {max_cloud_cover}% cloud cover')
            else:
                progress.update(search_progress, description=f'[red]Searching for Sentinel2 data between {start_date} and {end_date} and {max_cloud_cover}% cloud cover')
        search_result = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}" if start_date and end_date else None,
            query={
                "grid:code": {"eq": f"MGRS-{mgrs_id}"},
                "eo:cloud_cover": {"lte": max_cloud_cover},
            }
        )
        items = [itm.to_dict() for itm in search_result.items()]
        logger.debug(
            f'Found {len(items)} items in MGRS-{mgrs_id} between {start_date} and {end_date} and {max_cloud_cover}% cloud cover')

        if not items: #early exit
            start_date, end_date = expand_timerange(start_date=start_date, end_date=end_date, )
            continue

        if stop.is_set():break
        def process_item(it, mgrs_id, mid):
            props = it.get("properties", {})
            item_id = it.get("id", "")
            tile_info = get_tileinfo(it)
            dt_iso = props.get("datetime")
            if not dt_iso:
                return None

            return Candidate(
                id=item_id,
                time_ts = _iso_to_ts(dt_iso),
                ref_ts = mid,
                cloud_cover = _cloud_from_props(props),
                assets = it.get("assets", {}),
                grid = mgrs_id,
                nodata_coverage = 100 - tile_info["dataCoveragePercentage"],
                tile_geometry = shape(tile_info["tileGeometry"]),
                tile_data_geometry = shape(tile_info["tileDataGeometry"]),
                data_coverage= tile_info["dataCoveragePercentage"]
            )

        candidates = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_item, it, mgrs_id, mid) for it in items]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        cand = result
                        candidates.append(cand)
            except KeyboardInterrupt:
                for future in futures:
                    if not future.cancelled():
                        future.cancel()
                break
        #sort candidates
        scandidates = sorted(candidates, key=lambda c:-c.quality_score)
        #merge/unnion the polygons covering the valid pixels until the whole MGRS 100K tile is covered
        union = unary_union([x.tile_data_geometry for x in scandidates])
        coverage = int(floor(union.area/mgrs_poly.area*100))

        if coverage-prev_coverage == 0 and coverage>100:
            candidates = scandidates
            logger.debug(f'Found {len(candidates)} suitable candidates in {mgrs_id} between {start_date} and {end_date}')
            break
        # no suitable candidates found, enlarge the time interval by one week on both sides and try again
        prev_coverage = coverage
        start_date, end_date = expand_timerange(start_date=start_date, end_date=end_date, )
        if stop.is_set(): break

    # candidates were found, check if they should be pruned
    # prunning = optimisation in the sense that candidates with very similar spatial extent of data pixels are ignored
    # in this way a MGRS grid can be easily assembled

    if progress is not None and search_progress is not None:
        progress.remove_task(search_progress)

    if prune:
        pruned = []
        coverage = None
        for i, c in enumerate(candidates):
            if not pruned:
                coverage = mgrs_poly.intersection(c.tile_data_geometry)
                pruned.append(c)
            else:
                # new extends current coverage and intersects theoretical MGRS
                cov = coverage.union(c.tile_data_geometry).intersection(mgrs_poly)

                similarity_iou = coverage.intersection(cov).area / coverage.union(cov).area * 100
                similarity_hauss = sim(coverage, cov, coverage.hausdorff_distance(cov))
                similarity = (similarity_iou+similarity_hauss)*.5
                if similarity > 90:
                    continue
                if similarity == 100:
                    break
                pruned.append(c)
                coverage = cov
        return pruned

    return candidates




def fetch_s2_tiles(
    bbox=None,
    collection="sentinel-2-l1c",
    start_date=None,
    end_date=None,
    max_cloud_cover=None,
    progress = None,
    prune=None
):

    tiles = {}
    failed = {}
    stop = Event()
    ndone = 0

    client = pystac_client.Client.open(CATALOG_URL)
    mgrs_grids = generate_mgrs_tiles(bbox=bbox)
    mgrs_grids = ['21LYF']
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        jobs = dict()
        for grid_id in mgrs_grids:
            jobs_dict = dict(
                client=client, collection=collection, max_cloud_cover=max_cloud_cover,
                start_date=start_date, end_date=end_date,mgrs_id=grid_id, stop=stop,
                progress=progress, prune=prune
                )
            jobs[executor.submit(search, **jobs_dict)] = grid_id
        njobs = len(mgrs_grids)
        total_task = None
        if progress is not None:
            total_task = progress.add_task(
                description=f'[red]Going to search Sentinel2 data  in {njobs} MGRS 100K grids ', total=njobs)
        try:
            for future in concurrent.futures.as_completed(jobs):
                gid = jobs[future]
                try:
                    candidates = future.result()
                    tiles[gid] = candidates
                except Exception as e:
                    failed[gid] = e
                ndone += 1
                if progress is not None and total_task is not None:
                    progress.update(total_task,
                                    description=f'[red]Processed {ndone} MGRS grids',
                                    advance=1)
        except KeyboardInterrupt:
            stop.set()
            for future in jobs:
                if not future.cancelled():
                    future.cancel()

    for grid, err in failed.items():
        logger.error(f'Failed to search S2 tiles in {grid} because {err}')
    return tiles


if __name__ == "__main__":
    import asyncio
    import geopandas as gpd
    from rapida.util.setup_logger import setup_logger
    from rapida.components.landuse.search_utils.s2 import Sentinel2Item
    # logger.setLevel(logging.DEBUG)
    logger = setup_logger(level=logging.INFO)
    CATALOG_URL = "https://earth-search.aws.element84.com/v1"
    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    NIGERIA_BBOX = [6.0, 7.0, 8.0, 9.0]
    CHINA_BBOX = [100.0, 30.0, 110.0, 40.0]
    with Progress() as progress:
        bbox = BRAZIL_BBOX
        max_cloud_cover=3
        start_date = "2024-03-01"
        end_date = "2024-03-30"
        results = fetch_s2_tiles(bbox=bbox,
                                 start_date=start_date, end_date=end_date,
                                 max_cloud_cover=max_cloud_cover, progress=progress, prune=True)
        bands = ['B01', 'B02', 'B03', 'B04', 'B05']
        #bands = ['B01']

        for grid, candidates in results.items():
            try:
                logger.info(f'{grid}: {[c for c in candidates]}')
                s2i =  Sentinel2Item(mgrs_grid=grid, s2_tiles=candidates, root_folder='/tmp')

                asyncio.run(s2i.download(bands=bands, progress=progress, force=False))

            except KeyboardInterrupt:
                pass


