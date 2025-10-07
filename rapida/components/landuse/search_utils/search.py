
import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, datetime
import pystac_client
from dateutil import parser as dateparser
#from rapida.util.chunker import chunker
from rapida.components.landuse.search_utils.tiles import _cloud_from_props, _iso_to_ts, Candidate, get_tileinfo
from rapida.components.landuse.search_utils.zones import generate_mgrs_tiles
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
            start_date=None,end_date=None, mgrs_id=None,
            ):
    """

    :param client:
    :param collection:
    :param start_date:
    :param end_date:
    :param mgrs_id:
    :return:
    """

    nitems = 0
    min_data_coverage = 0
    while nitems < 1 and min_data_coverage <= 90:
        #print(f'Searching in {mgrs_id} between {start_date} and {end_date}')
        search_result = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}" if start_date and end_date else None,
            # query={
            #     "grid:code": {"eq": f"MGRS-{mgrs_id}"},
            #     "eo:cloud_cover": {"lte": 10},
            # },

            filter_lang="cql2-json",
            filter={
                "and": [
                    {"eq": [{"property": "grid:code"}, f"MGRS-{mgrs_id}"]},
                    # accept any of the common cloud fields, coalesced to a number
                    {"lte": [
                        {"coalesce": [
                            {"property": "eo:cloud_cover"},
                            {"property": "s2:cloud_coverage_assessment"},
                            {"property": "cloudyPixelPercentage"}
                        ]},
                        10
                    ]}
                ]
            },
        )

        items = [itm.to_dict() for itm in search_result.items()]

        start_date, end_date = expand_timerange(start_date=start_date, end_date=end_date,)
        nitems = len(items)

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
                tile_geometry = tile_info["tileGeometry"],
                tile_data_geometry = tile_info["tileDataGeometry"],
                data_coverage= tile_info["dataCoveragePercentage"]
            )


        candidates = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_item, it, mgrs_id) for it in items]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    cand = result
                    min_data_coverage = max(min_data_coverage, cand.data_coverage)
                    candidates.append(cand)
    if mgrs_id == '21LYE':
        print(mgrs_id, start_date, end_date, candidates[0].cloud_cover, min_data_coverage)
    return sorted(candidates, key=lambda c: (c.time_ts, c.cloud_cover))




def fetch_s2_tiles(
    client=None,
    collection="sentinel-2-l1c",
    mgrs_grids=None,
    start_date=None,
    end_date=None,
):

    tiles = {}
    failed = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        jobs = dict()
        for grid_id in mgrs_grids:
            jobs_dict = dict(
                client=client, collection=collection,
                start_date=start_date, end_date=end_date,mgrs_id=grid_id
                )
            jobs[executor.submit(search, **jobs_dict)] = grid_id
        try:
            for future in concurrent.futures.as_completed(jobs):
                gid = jobs[future]
                try:
                    candidates = future.result()
                    tiles[gid] = candidates
                except Exception as e:
                    failed[gid] = e
        except KeyboardInterrupt:
            for future in jobs:
                if not future.cancelled():future.cancel()

    return tiles


if __name__ == "__main__":
    #from rapida.util.setup_logger import setup_logger
    logger.setLevel(logging.DEBUG)
    # logger = setup_logger(level=logging.DEBUG)
    CATALOG_URL = "https://earth-search.aws.element84.com/v1"
    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    bbox = BRAZIL_BBOX
    start_date = "2024-03-01"
    end_date = "2024-03-30"
    client = pystac_client.Client.open(CATALOG_URL)
    grids = generate_mgrs_tiles(bbox=bbox)

    start_time = datetime.now()
    results = fetch_s2_tiles(client=client, mgrs_grids=grids, start_date=start_date, end_date=end_date)
    end_time = datetime.now()
    print(f"Elapsed time: {(end_time - start_time).total_seconds()} seconds")
    for grid in grids:
        print(grid, len(results[grid]), [(c.cloud_cover, c.data_coverage) for c in results[grid]])

