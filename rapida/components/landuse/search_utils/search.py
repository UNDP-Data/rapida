
import concurrent
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
from threading import Event
from rich.progress import Progress, TaskProgressColumn,BarColumn, TimeRemainingColumn, TextColumn


import logging
logger = logging.getLogger(__name__)







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
            stop:Event = None
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
    while True:
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

        if not items: #early exit
            start_date, end_date = expand_timerange(start_date=start_date, end_date=end_date, )
            continue

        if stop.is_set():break
        def process_item(it, mgrs_id):
            props = it.get("properties", {})
            item_id = it.get("id", "")
            tile_info = get_tileinfo(it)
            dt_iso = props.get("datetime")
            if not dt_iso:
                return None

            return Candidate(
                id=item_id,
                time_ts = _iso_to_ts(dt_iso),
                cloud_cover = _cloud_from_props(props),
                meta = {"hrefs": it.get("assets", {}), "grid": mgrs_id},
                nodata_coverage = 100 - tile_info["dataCoveragePercentage"],
                tile_geometry = shape(tile_info["tileGeometry"]),
                tile_data_geometry = shape(tile_info["tileDataGeometry"]),
                data_coverage= tile_info["dataCoveragePercentage"]
            )


        candidates = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_item, it, mgrs_id) for it in items]
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
        scandidates = sorted(candidates, key=lambda c: (abs(c.time_ts - mid), c.cloud_cover, -c.data_coverage))
        union = unary_union([x.tile_data_geometry for x in scandidates])
        coverage = int(floor(union.area/mgrs_poly.area*100))

        if coverage-prev_coverage == 0 and coverage>100:
            candidates = scandidates
            logger.debug(f'Found {len(candidates)} suitable candidates in {mgrs_id} between {start_date} and {end_date}')
            break
        prev_coverage = coverage
        start_date, end_date = expand_timerange(start_date=start_date, end_date=end_date, )
        if stop.is_set(): break

    return candidates




def fetch_s2_tiles(
    client=None,
    collection="sentinel-2-l1c",
    mgrs_grids=None,
    start_date=None,
    end_date=None,
):

    tiles = {}
    failed = {}
    stop = Event()
    ndone = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        with Progress() as progress:
            jobs = dict()
            for grid_id in mgrs_grids:
                jobs_dict = dict(
                    client=client, collection=collection,
                    start_date=start_date, end_date=end_date,mgrs_id=grid_id, stop=stop
                    )
                jobs[executor.submit(search, **jobs_dict)] = grid_id
            njobs = len(mgrs_grids)
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
                    progress.update(total_task,
                                    description=f'[red]Processed {ndone} MGRS grids',
                                    advance=1)

            except KeyboardInterrupt:
                stop.set()
                for future in jobs:
                    if not future.cancelled():
                        future.cancel()


    return tiles


if __name__ == "__main__":
    from rapida.util.setup_logger import setup_logger
    # logger.setLevel(logging.DEBUG)
    logger = setup_logger(level=logging.INFO)
    CATALOG_URL = "https://earth-search.aws.element84.com/v1"
    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    NIGERIA_BBOX = [6.0, 7.0, 8.0, 9.0]
    CHINA_BBOX = [100.0, 30.0, 110.0, 40.0]

    bbox = CHINA_BBOX

    start_date = "2024-03-01"
    end_date = "2024-03-30"
    client = pystac_client.Client.open(CATALOG_URL)
    grids = generate_mgrs_tiles(bbox=bbox)

    results = fetch_s2_tiles(client=client, mgrs_grids=grids, start_date=start_date, end_date=end_date)
    end_time = datetime.now()

    for grid, candiadtes in results.items():
        r = results[grid]
        print(grid, [c for c in r])




