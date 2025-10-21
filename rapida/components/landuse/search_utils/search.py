
import concurrent
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, datetime

import geopandas
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
                    description=f'[red]Searching for Sentinel2 data between {start_date} and {end_date} and {max_cloud_cover}% cloud cover',
                total=None, start=False)
            else:
                progress.update(search_progress, description=f'[red]Searching for Sentinel2 data between {start_date} and {end_date} and {max_cloud_cover}% cloud cover',
                                total=None,)
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

        candidates = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_item, itm, mgrs_id, mid, mgrs_poly, crs) for itm in items]
            try:
                for future in as_completed(futures):
                    cand = future.result()
                    if cand:
                        candidates.append(cand)
            except KeyboardInterrupt:
                for future in futures:
                    if not future.cancelled():
                        future.cancel()
                break
        #sort candidates
        scandidates = sorted(candidates, key=lambda c:-c.quality_score)

        #merge/union the polygons covering the valid pixels until the whole MGRS 100K tile is covered
        union = unary_union([x.tile_data_geometry for x in scandidates])
        coverage = int(floor(union.area/mgrs_poly.area*100))

        #if coverage-prev_coverage == 0 and coverage>100:
        if coverage>100:
            candidates = scandidates
            logger.debug(f'Found {len(candidates)} suitable candidates in {mgrs_id} between {start_date} and {end_date}')
            break
        # no suitable candidates found, enlarge the time interval by one week on both sides and try again
        prev_coverage = coverage
        start_date, end_date = expand_timerange(start_date=start_date, end_date=end_date, )
        if stop.is_set(): break

    # candidates were found, check if they should be pruned
    # pruning = optimisation in the sense that candidates with very similar spatial extent of data pixels are ignored
    # in this way a MGRS grid can be easily assembled

    if progress is not None and search_progress is not None:
        progress.remove_task(search_progress)

    if prune:
        pruned = []
        coverage = None
        prev_coverage_score_perc = None
        for i, c in enumerate(candidates):


            if not pruned:
                coverage = mgrs_poly.intersection(c.tile_data_geometry)
                pruned.append(c)
                # gdf = gpd.GeoDataFrame(geometry=[c.tile_data_geometry], crs=crs)
                # gdf.to_file(f'/tmp/{mgrs_id}/coverage_{i}.fgb')
                prev_coverage_score_perc = coverage.area/mgrs_poly.area*100
            else:

                # new extends current coverage and intersects theoretical MGRS
                #cov_poly = c.tile_data_geometry.intersection(mgrs_poly).union(coverage)
                covc = c.tile_data_geometry.intersection(mgrs_poly)

                cov_poly = coverage.union(covc)

                r = covc.area/coverage.area*100
                cov_score_perc = cov_poly.area/mgrs_poly.area*100

                # the similarity is more complicated
                similarity = coverage.intersection(cov_poly).area / coverage.union(cov_poly).area * 100
                # similarity_hauss = sim(coverage, cov_poly, coverage.hausdorff_distance(cov_poly))
                # similarity = (similarity+similarity_hauss)*.5
                r = cov_score_perc-prev_coverage_score_perc


                # gdf = gpd.GeoDataFrame(geometry=[c.tile_data_geometry], crs=crs)
                # gdf.to_file(f'/tmp/{mgrs_id}/coverage_{i}.fgb')
                if similarity > 90 and r == 0:
                    logger.debug(f'Pruning {c} because of high similarity {similarity} with coverage')
                    continue
                if cov_score_perc >= 100 or r>0 :
                    pruned.append(c)
                    break
                #print(i, pruned, c, similarity, cov_score_perc,r)
                coverage = cov_poly
                prev_coverage_score_perc = cov_score_perc


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
    mgrs_grids = list(generate_mgrs_tiles(bbox=bbox))
    #mgrs_grids = mgrs_grids[1:2]
    #mgrs_grids = ['21LYF']
    #mgrs_grids = ['21LXD']
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
                    cands = future.result()
                    tiles[gid] = cands
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
    from osgeo import gdal

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
        #bands = ['B03']
        band_files = {}
        for grid, candidates in results.items():
            try:
                #logger.info(f'{grid}: {[c for c in candidates]}')
                s2i =  Sentinel2Item(mgrs_grid=grid, s2_tiles=candidates, root_folder='/tmp')
                downloaded = s2i.download(bands=bands, progress=progress, force=False)
                for bname, bfile in downloaded.items():
                    if not bname in band_files:
                        band_files[bname] = []
                    band_files[bname].append(bfile)


            except KeyboardInterrupt:
                pass

        for band, bfiles in band_files.items():
            with gdal.BuildVRT(f'/tmp/{band}.vrt',bfiles) as vrt_ds:
                vrt_ds.GetRasterBand(1).ComputeStatistics(approx_ok=True)
