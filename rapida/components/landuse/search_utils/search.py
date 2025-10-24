from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, datetime
import pystac_client
from dateutil import parser as dateparser
from  math import floor
from rapida.components.landuse.search_utils.mgrstiles import (_cloud_from_props, _iso_to_ts,
                                                              Candidate, get_tileinfo, midpoint_from_range
                                                              )
from rapida.components.landuse.search_utils.mgrsconv import mgrs_100k_tiles_for_bbox
from shapely.geometry import shape
from shapely.ops import unary_union
from threading import Event
from rich.progress import Progress
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
            start_date=None,end_date=None, mgrs_id=None, mgrs_poly=None, crs=None, max_cloud_cover=10,
            stop:Event = None, progress:Progress=None, prune=False
            ) -> list[Candidate]:
    """
        Search  Sentinel 2 L1C imagery usinng STAC API in a specific location corresponding to an MGRS 100K grid square
        in a specific time interval and with a specified  max cloud cover.
        This function is adaptive, in the sense it expands automatically the time interval up to 90 days from start date
        in case no items have been found. Additionally, it is greedy and considers images with partial data coverage
        because these can be STICHED together.

        :param client: stac client instance
        :param collection: str, S2 colenction to search
        :param start_date: str,
        :param end_date: str,
        :param mgrs_id: 100K MGRS Grid id (ex 21LYE)
        :param mgrs_poly: shapely polygon in UTM coords, used to compute coverage
        :param crs: the (UTM) projection of the MGRS tile
        :param max_cloud_cover: int
        :param stop: event  to force early exit
        :param progress: progress bart
        :param prune:bool. If true the candidates wil be trimmed/pruned at the point the full coverage is truly achieved
        This is because the search is greedy and return more items. Otherwise all these items/images would be downloaded
        :return: list of candidates or an empty list
    """
    mid = midpoint_from_range(range_str=f"{start_date}/{end_date}" if start_date and end_date else None)
    mid_date = datetime.fromtimestamp(mid).date()
    niter = 0
    search_progress = None
    cloud_cover = max_cloud_cover
    tile_candidates: list[Candidate] = []
    while True:
        # this is because the start tiles on every row in every UTM zone do not contain data
        if (mid_date-datetime.fromisoformat(start_date).date()).days > 90:
            break
        if progress is not None:
            if search_progress is None:
                search_progress = progress.add_task(
                    description=f'[red]Searching for Sentinel2 data in MGRS-{mgrs_id} between {start_date} and {end_date} and {cloud_cover}% cloud cover',
                total=None, start=False)
            else:
                progress.update(search_progress, description=f'[red]Searching for Sentinel2 data  in MGRS-{mgrs_id} between {start_date} and {end_date} and {cloud_cover}% cloud cover',
                                total=None,)
        search_result = client.search(
            collections=[collection],
            datetime=f"{start_date}/{end_date}" if start_date and end_date else None,
            query={
                "grid:code": {"eq": f"MGRS-{mgrs_id}"},
                "eo:cloud_cover": {"lte": cloud_cover},
            }
        )
        items = [itm.to_dict() for itm in search_result.items()]
        logger.debug(
            f'Found {len(items)} items in MGRS-{mgrs_id} between {start_date} and {end_date} and {cloud_cover}% cloud cover')

        if not items: #early exit
            start_date, end_date = expand_timerange(start_date=start_date, end_date=end_date)
            continue

        if stop.is_set():break
        def process_item(it, mgrs_id, ref_ts, mgrs_poly, mgrs_crs):
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
                assets = it.get("assets", {}),
                grid = mgrs_id,
                nodata_coverage = 100 - tile_info["dataCoveragePercentage"],
                tile_geometry = shape(tile_info["tileGeometry"]),
                tile_data_geometry = shape(tile_info["tileDataGeometry"]),
                data_coverage= tile_info["dataCoveragePercentage"],
                ref_ts = ref_ts,
                mgrs_geometry = mgrs_poly,
                mgrs_crs = mgrs_crs
            )


        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_item, itm, mgrs_id, mid, mgrs_poly, crs) for itm in items]
            try:
                for future in as_completed(futures):
                    cand = future.result()
                    if cand:
                        tile_candidates.append(cand)
            except KeyboardInterrupt:
                for future in futures:
                    if not future.cancelled():
                        future.cancel()
                break
        #sort candidates
        scandidates = sorted(tile_candidates, key=lambda c:-c.quality_score)

        #merge/union the polygons covering the valid pixels until the whole MGRS 100K tile is covered
        union = unary_union([x.tile_data_geometry for x in scandidates])
        coverage = int(floor(union.area/mgrs_poly.area*100))
        # exit here because the whole tile can be covered
        if coverage>100:
            tile_candidates = scandidates
            logger.debug(f'Found {len(tile_candidates)} suitable candidates in {mgrs_id} between {start_date} and {end_date}')
            break
        # no suitable candidates found, enlarge the time interval by one week on both sides and try again
        start_date, end_date = expand_timerange(start_date=start_date, end_date=end_date, )
        if stop.is_set(): break
        niter+=1




    # candidates were found, check if they should be pruned
    # pruning = optimisation in the sense that candidates with very similar spatial extent of data pixels are ignored
    # in this way a MGRS grid can be easily assembled

    if progress is not None and search_progress is not None:
        progress.remove_task(search_progress)
    if prune and tile_candidates:
        pruned = []
        coverage = None # in percent
        i = 0
        cov_poly = None # shapely polygon
        while True:
            c = tile_candidates[i]
            i+=1
            if not coverage:
                # first/best, so we keep it but still try to exit
                cov_poly = mgrs_poly.intersection(c.tile_data_geometry)
                coverage = cov_poly.area / mgrs_poly.area * 100
                pruned.append(c)
                if round(coverage) >= 100:
                    break
            else:
                # if any other candidate is toi similar skip it because it will bring little value
                similarity = cov_poly.intersection(c.tile_data_geometry).area / cov_poly.union(c.tile_data_geometry).area * 100
                if similarity > 50:continue
                # adjust polygon coverage and recompute new coverage in perc. Try to exit
                cov_poly = c.tile_data_geometry.intersection(mgrs_poly).union(cov_poly)
                coverage = cov_poly.area/mgrs_poly.area*100
                pruned.append(c)
                if round(coverage) >= 100: # round is for 99.99..etc
                    break

        return pruned

    return tile_candidates




def fetch_s2_tiles(
    stac_url=None, bbox=None, collection="sentinel-2-l1c",
    start_date=None, end_date=None, max_cloud_cover=None,
    progress = None, prune=True
) -> dict[str:list[Candidate]]:
    """
    Fetch Sentinel2 imagery from stac_url between start and end dates with max_cloud_cover
    in a specific geographic area defined by the bbox .
    THe function operates concurrently by identifying all MGRS 100K tiles that intersect


    :param stac_url: str, url of STAC service
    :param bbox: geographic bbox, (lon_min, lat_min, lon_max, lat_max)
    :param collection: str, S2 collection
    :param start_date:str
    :param end_date:str
    :param max_cloud_cover: int
    :param progress:rich progress
    :param prune:bool
    :return: dict of identified MGRS 100K grid that intersect the bbox and the list of candidate image tiles
    """
    tiles = {}
    failed = {}
    stop = Event()
    ndone = 0

    client = pystac_client.Client.open(stac_url)
    mgrs_grids = mgrs_100k_tiles_for_bbox(*bbox) # avoid data
    #mgrs_grids = {'36NWF':mgrs_grids['36NWF']}

    with ThreadPoolExecutor(max_workers=5) as executor:

        jobs = dict()
        for grid_id, (mgrs_poly, crs) in mgrs_grids.items():
            jobs_dict = dict(
                client=client, collection=collection, max_cloud_cover=max_cloud_cover,
                start_date=start_date, end_date=end_date,mgrs_id=grid_id, mgrs_poly=mgrs_poly, crs=crs, stop=stop,
                progress=progress, prune=prune
                )
            jobs[executor.submit(search, **jobs_dict)] = grid_id
        njobs = len(mgrs_grids)
        total_task = None
        if progress is not None:
            total_task = progress.add_task(
                description=f'[red]Going to search Sentinel2 data  in {njobs} MGRS 100K grids ', total=njobs)
        try:
            for future in as_completed(jobs):
                gid = jobs[future]
                try:
                    cands = future.result()
                    # tiles at edge are smaller and only one of them holds data that extends into the neighbour zone
                    # so the ones that have no imagery (no candidates were foud) are discarded
                    if cands:
                        tiles[gid] = cands
                except Exception as e:
                    failed[gid] = e
                ndone += 1
                if progress is not None and total_task is not None:
                    progress.update(total_task,
                                    description=f'[red]Processed {gid} MGRS grid',
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

    from rapida.util.setup_logger import setup_logger
    from rapida.components.landuse.search_utils.s2item import Sentinel2Item
    # logger.setLevel(logging.DEBUG)
    logger = setup_logger(level=logging.INFO)
    CATALOG_URL = "https://earth-search.aws.element84.com/v1"
    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    NIGERIA_BBOX = [6.0, 7.0, 8.0, 9.0]
    CHINA_BBOX = [100.0, 30.0, 110.0, 40.0]
    UGAKEN_BBOX = 33.280908600000004, -1.154692598710874, 36.59833289999998, 2.650255597059965


    with Progress() as progress:
        bbox=UGAKEN_BBOX
        max_cloud_cover=3
        start_date = "2024-03-01"
        end_date = "2024-03-30"
        results = fetch_s2_tiles(bbox=bbox,stac_url=CATALOG_URL,
                                 start_date=start_date, end_date=end_date,
                                 max_cloud_cover=max_cloud_cover, progress=progress, prune=True)


        bands = ['B01', 'B02', 'B03', 'B04', 'B05']
        bands = ['B03']
        band_files = {}

        for grid, candidates in results.items():
            try:
                logger.info(f'{grid}: {[c for c in candidates]}')
                s2i =  Sentinel2Item(mgrs_grid=grid, s2_tiles=candidates, root_folder='/tmp')
                downloaded = s2i.download(bands=bands, progress=progress, force=False)
                for bname, bfile in downloaded.items():
                    if not bname in band_files:
                        band_files[bname] = []
                    band_files[bname].append(bfile)


            except KeyboardInterrupt:
                pass

        for band, bfiles in band_files.items():
             for fl in bfiles:
                 
                 print(band, fl)
            # with gdal.BuildVRT(f'/tmp/{band}.vrt',bfiles) as vrt_ds:
            #     vrt_ds.GetRasterBand(1).ComputeStatistics(approx_ok=True)
