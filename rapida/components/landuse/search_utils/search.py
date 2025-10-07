import asyncio
import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import timedelta, datetime

import pystac_client
from dateutil import parser as dateparser

from rapida.util.chunker import chunker
from rapida.components.landuse.search_utils.tiles import _cloud_from_props, _iso_to_ts, Candidate, get_tileinfo
from rapida.components.landuse.search_utils.zones import generate_mgrs_tiles


executor = ThreadPoolExecutor(max_workers=10)


def expand_timerange(start_date: str, end_date: str, days: int = 7) -> tuple[str, str]:
    start_dt = dateparser.parse(start_date)
    end_dt = dateparser.parse(end_date)
    today = datetime.today()

    expanded_start = start_dt - timedelta(days=days)
    expanded_end = end_dt + timedelta(days=days)

    if expanded_end > today:
        expanded_end = today

    return expanded_start.date().isoformat(), expanded_end.date().isoformat()


def search(
    client=None,
    bbox=None,
    collection="sentinel-2-l1c",
    limit=None,
    start_date=None,
    end_date=None,
    mgrs_id=None,
):
    stac = client.search(
        collections=[collection],
        bbox=bbox,
        max_items=limit,
        datetime=f"{start_date}/{end_date}" if start_date and end_date else None,
        query={"grid:code": {"eq": f"MGRS-{mgrs_id}"}},
    )
    items = [itm.to_dict() for itm in stac.items()]

    def process_item(it):
        props = it.get("properties", {})
        item_id = it.get("id", "")
        tile_info = get_tileinfo(it)
        dt_iso = props.get("datetime")
        if not dt_iso:
            return None

        cand = Candidate(
            id=item_id,
            time_ts=_iso_to_ts(dt_iso),
            cloud=_cloud_from_props(props),
            meta={"hrefs": it.get("assets", {}), "grid": mgrs_id},
            nodata=100 - tile_info["dataCoveragePercentage"],
            tile_geometry=tile_info["tileGeometry"],
            tile_data_geometry=tile_info["tileDataGeometry"],
        )
        return mgrs_id, cand

    tiles = defaultdict(list)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_item, it) for it in items]
        for future in as_completed(futures):
            result = future.result()
            if result:
                grid, cand = result
                tiles[grid].append(cand)

    for tid in tiles:
        tiles[tid].sort(key=lambda c: c.time_ts)

    return tiles


async def main(
    bbox=None,
    client=None,
    collection="sentinel-2-l1c",
    start_date=None,
    end_date=None,
):
    grids = generate_mgrs_tiles(bbox=bbox)
    group_results = {}
    loop = asyncio.get_event_loop()

    # Expand the time range
    # expanded_start, expanded_end = expand_timerange(start_date, end_date)

    for grid_group in chunker(grids, size=5):
        tasks = [
            loop.run_in_executor(
                executor,
                search,
                client,
                bbox,
                collection,
                None,
                start_date,
                end_date,
                grid_id,
            )
            for grid_id in grid_group
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                raise r
            tiles, gid = r
            group_results[gid] = tiles

    return group_results


def main1(
    bbox=None,
    client=None,
    collection="sentinel-2-l1c",
    start_date=None,
    end_date=None,
):
    grids = generate_mgrs_tiles(bbox=bbox)
    tiles = {}
    failed = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        jobs = {executor.submit(search, client, bbox, collection, None, start_date, end_date, grid_id): grid_id for grid_id in grids}
        for future in concurrent.futures.as_completed(jobs):
            gid = jobs[future]
            try:
                tiles = future.result()
            except:
                failed[gid] = future.exception()
    for gid, error in failed.items():
        print(f"Error for grid {gid}: {error}")
    return tiles


if __name__ == "__main__":
    CATALOG_URL = "https://earth-search.aws.element84.com/v1"
    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    bbox = BRAZIL_BBOX
    start_date = "2024-01-01"
    end_date = "2024-06-30"
    client = pystac_client.Client.open(CATALOG_URL)

    start_time = datetime.now()
    # results = asyncio.run(
    #     main(bbox, client, start_date=start_date, end_date=end_date, collection="sentinel-2-l1c")
    # )
    results = main1(bbox, client, start_date=start_date, end_date=end_date, collection="sentinel-2-l1c")
    end_time = datetime.now()

    print(f"Elapsed time: {(end_time - start_time).total_seconds()} seconds")
    print(results["21LZD"])
