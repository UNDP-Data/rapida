import asyncio
import datetime
import threading

from rasterio.windows import bounds, from_bounds

from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed

import numpy as np

from rasterio.merge import merge
import os.path
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rich.progress import Progress
import aiohttp
import aiofiles
import random
from affine import Affine
from rapida.components.landuse.constants import SENTINEL2_ASSET_MAP, SENTINEL2_BAND_MAP
from rapida.components.landuse.search_utils.mgrstiles import Candidate
from rapida.components.landuse.search_utils.mgrsconv import _parse_mgrs_100k
from typing import List, Tuple
import math
import logging

logger = logging.getLogger(__name__)


def compute_offsets(remote_size: int, nchunks: int) -> List[Tuple[int, int]]:
    """
    Split [0, remote_size) into <= nchunks contiguous, gap-free, non-overlapping ranges.
    Final chunk absorbs the remainder.
    """
    if nchunks <= 0:
        raise ValueError("nchunks must be >= 1")
    if remote_size <= 0:
        return [(0, 0)]

    # ceil() ensures we never create more than `nchunks` tasks
    chunk_size = math.ceil(remote_size / nchunks)
    offsets: List[Tuple[int, int]] = []
    start = 0
    while start < remote_size:
        end = min(start + chunk_size, remote_size)
        offsets.append((start, end))
        start = end

    # # Sanity checks
    # assert offsets[0][0] == 0
    # assert offsets[-1][1] == remote_size
    # assert all(offsets[i][1] == offsets[i+1][0] for i in range(len(offsets)-1))
    return offsets
class Sentinel2Item:
    """
    A sophisticated Sentinel 2 utility image that aligns perfectly with MGRS 100K grid tiles.
    While Sentinel 2 imagery is organized into tiles, these only followS roughly the UTM MGRS 100K grid.
    Additionally, Sentinel 2 images can come incomplete in a variety of data coverages. For this reason the
    Sentinel2Item class takes several S2 images and combines spatially disjoint tiles into one ideal MGRS 100K tile
    perfectly aligned with 100K MGRS grid
    """

    def __init__(self, mgrs_grid:str=None, s2_tiles:List[Candidate] = None, root_folder:str = None, concurrent_bands=50):

        self.utm_zone, self.mgrs_band, self.mgrs_grid_letters = _parse_mgrs_100k(grid_id=mgrs_grid)
        self.mgrs_grid = mgrs_grid
        self.s2_tiles = sorted(s2_tiles, key=lambda c:-c.quality_score)
        self.root_folder = os.path.join(os.path.abspath(root_folder), self.mgrs_grid)
        os.makedirs(self.root_folder, exist_ok=True)
        self.__urls__ = {}
        self.__files__ = {}
        self.__bands__ = {}
        self._prepare_urls_()
        for band in self.bands:
            band_file = os.path.join(self.root_folder, f'{band}.tif')
            self[band] = band_file
        self.semaphore = asyncio.Semaphore(concurrent_bands)

    def _s3_to_http(self, url):
        """
        Convert s3 protocol to http protocol

        :param url: URL starts with s3://
        :return http url
        """
        if url.startswith('s3://'):
            s3_path = url
            bucket = s3_path[5:].split('/')[0]
            # NOTE: This is a workaround to earth search issue where the data points to a different bucket.
            # See issue: https://github.com/Element84/earth-search/issues/3
            if bucket != 'sentinel-s2-l1c':
                bucket = 'sentinel-s2-l1c'
            object_name = '/'.join(s3_path[5:].split('/')[1:])
            return 'https://{0}.s3.amazonaws.com/{1}'.format(bucket, object_name)
        else:
            return url


    def _prepare_urls_(self):
        if not self.__urls__:
            for asset_name, asset_band in SENTINEL2_ASSET_MAP.items():
                for cand in self.s2_tiles:
                    asset_s3_uri = cand.assets[asset_name]['href']
                    asset_html_uri = self._s3_to_http(asset_s3_uri)
                    if asset_band in self.__urls__:
                        self.__urls__[asset_band].append(asset_html_uri)
                    else:
                        self.__urls__[asset_band] = [asset_html_uri]
    @property
    def urls(self):
        return self.__urls__

    @property
    def files(self):
        return self.__files__

    @property
    def bands(self):
        return sorted(self.__urls__.keys())


    def __setitem__(self, key, value):
        assert key in self.bands, f'Sentinel2 band "{key}" is invalid. Valid bands are {"".join(self.bands)}'
        self.__bands__[key] = value

    def __getitem__(self, key):
        if not key in self.bands:
            raise ValueError(f'Sentinel2 band "{key}" is invalid. Valid bands are {"".join(self.bands)}')
        return self.__bands__.get(key, None)



    async def __fetch_range__(
            self, session: aiohttp.ClientSession, url: str, start: int, end: int,
            max_retries: int = 4, base_backoff: float = 0.25
    ) -> bytes:

        headers = {"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"}
        expected = end - start
        attempt = 0

        while True:
            try:
                async with session.get(url, headers=headers) as resp:
                    if resp.status not in (200, 206):
                        resp.raise_for_status()
                    data = await resp.read()
                    if len(data) != expected:
                        raise IOError(f"Expected {expected}B, got {len(data)}B for {start}-{end}")
                    return data

            except (aiohttp.ClientOSError,
                    aiohttp.ServerDisconnectedError,
                    aiohttp.ClientPayloadError,
                    asyncio.TimeoutError) as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                delay = base_backoff * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                logger.warning(
                    f"Range {start}-{end} failed ({type(e).__name__}: {e}); retry {attempt}/{max_retries} in {delay:.2f}s")
                await asyncio.sleep(delay)

    async def __download_file__(self, session: aiohttp.ClientSession = None,
                                url: str = None, dst: str = None, nchunks: int = 5, force=False):

        async with self.semaphore:
            # 1) Discover size (HEAD)
            async with session.head(url) as head_resp:
                head_resp.raise_for_status()
                cl = head_resp.headers.get("content-length")
                if cl is None:
                    raise ValueError("No content-length in response headers")
                remote_content_length = int(cl)

            # 1.5) early return if file exists, seems to be of identical size and force=False
            if os.path.exists(dst):
                if os.path.getsize(dst) == remote_content_length and not force:
                    logger.debug(f'Reusing {dst}')
                    return dst

            # 2) Launch chunk range tasks
            offsets = compute_offsets(remote_content_length, nchunks)
            tasks: list[asyncio.Task] = []
            for start_offset, end_offset in offsets:
                t = asyncio.create_task(
                    self.__fetch_range__(session=session, url=url,
                                         start=start_offset, end=end_offset),
                    name=f"{start_offset}-{end_offset}"
                )
                tasks.append(t)





            folder = os.path.dirname(dst)
            os.makedirs(folder, exist_ok=True)


            # 3) Write chunks as they finish
            async with aiofiles.open(dst, "wb") as local_file:
                # # Optional: pre-size to avoid sparse edge-cases
                # await local_file.truncate(remote_content_length)

                try:
                    pending = set(tasks)
                    while pending:
                        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                        for done_task in done:
                            soff, eoff = map(int, done_task.get_name().split("-"))
                            try:
                                data = await done_task  # may raise
                                await local_file.seek(soff)
                                await local_file.write(data)
                                logger.debug(f"Saved {len(data)}B to {dst} @ [{soff}-{eoff})")

                            except asyncio.CancelledError:
                                # We are cancelling: stop siblings, **no error logs** here.
                                for t in pending:
                                    t.cancel()
                                await asyncio.gather(*pending, return_exceptions=True)
                                raise

                            except Exception as e:
                                # Distinguish shutdown-induced network noise from real errors.
                                shutting_down_noise = (
                                        isinstance(e, RuntimeError) and "Connection closed" in str(e)
                                )
                                # Cancel siblings either way.
                                for t in pending:
                                    t.cancel()
                                await asyncio.gather(*pending, return_exceptions=True)

                                if shutting_down_noise:
                                    # Convert to cancellation so upper layer handles it; no log here.
                                    raise asyncio.CancelledError from e
                                else:
                                    # Real failure: report and propagate.
                                    logger.error(f"Failed to download bytes {soff}-{eoff} because {e}")
                                    raise

                finally:
                    # Ensure all tasks are drained exactly once
                    await asyncio.gather(*tasks, return_exceptions=True)

            # 4) Verify only if we weren’t cancelled
            assert os.path.exists(dst), f"{dst} does not exist"
            assert os.path.getsize(dst) == remote_content_length, f"{dst} was NOT downloaded successfully"
            return dst


    async def __download_band__(self, band:str=None, session=None,
                                progress=None, progress_task=None, force=False,
                                mosaic=True):
        """
        Download files that will be sued to make a single band mosaic
        :param band:
        :param session:
        :param progress:
        :param progress_task:
        :param force
        :param: mosaic
        :return:
        """
        band_file = self[band]


        band_urls = self.urls[band]
        tasks = []
        positions = {}
        for url in band_urls:
            chunk = url.split(self.mgrs_grid_letters)[-1]
            year, month, day, *rest = chunk[1:].split('/')
            date = datetime.date(year=int(year), month=int(month), day=int(day))
            position = [c.date for c in self.s2_tiles].index(date)
            original_file_name = '_'.join(rest)
            new_file_name = f"{date.strftime('%Y%m%d')}_{original_file_name}"
            file_path = os.path.join(self.root_folder,band,new_file_name )
            positions[file_path] = position
            tasks.append(
                asyncio.create_task(
                    self.__download_file__(session=session, url=url, dst=file_path, force=force),
                    name=new_file_name
                )
            )
        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for done_task in done:
                fname = done_task.get_name()

                try:
                    downloaded_file = await done_task  # already finished; get result or raise
                    if not band in self.__files__:
                        self.__files__[band] = []
                    position = positions[downloaded_file]
                    #self.__files__[band][position] = downloaded_file
                    self.__files__[band].insert(position, downloaded_file)

                    if progress is not None and progress_task is not None:
                        progress.update(progress_task,
                                        description=f'[red]Downloaded file {fname} band {band} in MGRS grid {self.mgrs_grid}',
                                        advance=1)
                except asyncio.CancelledError as ce:
                    if tasks:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                except Exception as e:
                    logger.error(f'Failed to download {band} because {e}')
                    raise
        self.__files__[band] = tuple(self.__files__[band])
        if progress is not None and progress_task is not None:
            progress.update(progress_task,
                            description=f'[red]Downloaded band {band} in MGRS grid {self.mgrs_grid}')

        if mosaic:
            self.__mosaic__(band=band, progress=progress)
        return self[band]



    async def __download__(self, bands: List[str] = None, progress: Progress = None,
                       connect_timeout=25, read_timeout=900, force=False):
        """
        Download  one or more Sentinel2 bands
        :param bands:
        :param progress:
        :param connect_timeout:
        :param read_timeout:
        :return:
        """

        nfiles = len(self.s2_tiles) * len(bands)
        timeout = aiohttp.ClientTimeout(total=None, connect=connect_timeout, sock_connect=read_timeout)
        downloaded_bands = {}
        progress_task = None
        # Mildly tuned connector for parallel ranged GETs
        connector = aiohttp.TCPConnector(
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
            # You can set limits here if you like; leaving defaults is fine too.
            # limit_per_host=len(bands) or a small constant; omit if unsure.
        )

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            band_tasks: set[asyncio.Task] = set()


            if progress is not None:
                progress_task = progress.add_task(
                    description=f"[red]Downloading Sentinel-2 for {self.mgrs_grid}",
                    total=nfiles
                )

            try:

                # Schedule all band downloads
                for band in bands:
                    assert band in self.bands, (
                        f'Sentinel2 band "{band}" is invalid. Valid bands are {"".join(self.bands)}'
                    )


                    t = asyncio.create_task(
                        self.__download_band__(band=band, session=session,
                                               progress=progress, progress_task=progress_task, force=force),
                        name=band
                    )
                    band_tasks.add(t)

                # Fail-fast loop: if any band task errors, cancel the rest
                while band_tasks:
                    done, band_tasks = await asyncio.wait(
                        band_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for dt in done:
                        band = dt.get_name()
                        try:
                            downloaded_band = await dt  # consume result or raise
                            downloaded_bands[band] = downloaded_band
                        except Exception as e:

                            logger.error(f"Band {band} failed: {e}")
                            # Convert to cancellation so the outer handler centralizes cleanup
                            raise asyncio.CancelledError() from e

            except asyncio.CancelledError:
                # Centralized shutdown: cancel remaining bands and drain once
                logger.info(f"Cancelling download for {self.mgrs_grid}")
                for t in band_tasks:
                    t.cancel()
                await asyncio.gather(*band_tasks, return_exceptions=True)
                raise
            finally:
                # Ensure all tasks are drained exactly once
                await asyncio.gather(*band_tasks, return_exceptions=True)
                for t in band_tasks:
                    if t.done():
                        logger.info(f'{t.get_name()} is done')
                if progress is not None and progress_task is not None:
                    progress.remove_task(progress_task)
                return downloaded_bands

    def __mosaic__(self, band: str = None, progress=None):

        band_file = self[band]
        if os.path.exists(band_file):
            return

        files = self.files[band]

        mgrs_poly = self.s2_tiles[0].mgrs_geometry
        xmin, ymin, xmax, ymax = mgrs_poly.bounds
        asset_name = SENTINEL2_BAND_MAP[band]
        asset_def = self.s2_tiles[0].assets[asset_name]
        no_data = asset_def['raster:bands'][0]['nodata']
        res = asset_def['raster:bands'][0]['spatial_resolution']
        transform = Affine.from_gdal(c=xmin,a=res, b=0, f=ymax, d=0, e=-res)
        width = math.floor((xmax-xmin)/res)
        height = math.floor((ymin-ymax)/-res)
        best = files[0]
        src_datasets = []
        vrt_datasets = []
        try:

            with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=1024, OPJ_NUM_THREADS="ALL_CPUS", USE_TILE_AS_BLOCK="YES"):
                with rasterio.open(best) as src:
                    mgrs_profile = src.profile
                    mgrs_profile.update(dict(
                        transform=transform,
                        width=width,
                        height=height,
                        tiled=True,
                        compress='ZSTD',
                        bigtiff="IF_SAFER",
                        driver="GTiff",
                        nodata=no_data


                    ))
                    # Build VRTs for all fallback inputs aligned to ref grid
                    vrt_opts = dict(
                        crs=src.crs,
                        transform=src.transform,
                        width=src.width,
                        height=src.height,
                        resampling=Resampling.nearest,
                    )

                    # Open all sources, keep VRTs for 2..N
                    src_datasets = [src]  # keep ref for lifetime
                    vrt_datasets = []
                    for p in files[1:]:
                        s = rasterio.open(p)
                        src_datasets.append(s)
                        vrt_datasets.append(WarpedVRT(s, **vrt_opts))


                    with rasterio.open(band_file, "w", **mgrs_profile) as dst:
                        wins = {block: win for block, win in dst.block_windows()}
                        progress_task = None
                        if progress is not None:
                            progress_task = progress.add_task(
                                description=f"[red]Merging Sentinel-2 for {self.mgrs_grid}",
                                total=len(wins)
                            )
                        for block, window in wins.items():
                            # __fill_window__(window=window,src_ds=src,vrt_datasets=vrt_datasets,no_data=no_data, dst_ds=dst)
                            bbox = bounds(window=window, transform=dst.transform)
                            src_win = from_bounds(*bbox, transform=src.transform).round()

                            # 1) Read dominant band as ndarray
                            out = src.read(1, window=src_win)

                            # 2) Build holes mask (where ref is nodata)
                            holes = (out == no_data)

                            if not holes.any():
                                dst.write(out, 1, window=window)
                                if progress is not None and progress_task is not None:
                                    progress.update(progress_task,
                                                    description=f'[red]Merged image block in MGRS grid {self.mgrs_grid}',
                                                    advance=1)

                                continue

                            # 3) Fill from each fallback VRT in priority order
                            for v in vrt_datasets:
                                if not holes.any():
                                    break  # short-circuit: all filled
                                arr = v.read(1, window=src_win)  # ndarray
                                valid = (arr != no_data)  # pixels we can take from fallback
                                take = holes & valid  # only fill holes with valid data

                                if take.any():
                                    out[take] = arr[take]  # in-place update
                                    holes[take] = False  # those holes are now filled

                            # 4) Any remaining holes -> write as nodata, should not happen
                            if holes.any():
                                logger.warning(f'Some pixels are still empty in {self.mgrs_grid}-{window.row_off, window.col_off, window.height, window.width}')
                                out[holes] = no_data

                            dst.write(out, 1, window=window)
                            if progress is not None and progress_task is not None:
                                progress.update(progress_task,
                                                description=f'[red]Processed image block in  MGRS grid {self.mgrs_grid}',
                                                advance=1)

            self[band] = band_file

        finally:
            for vrt in vrt_datasets:vrt.close()
            for s in src_datasets[1:]:s.close()
            if progress is not None and progress_task is not None:
                progress.remove_task(progress_task)


    def __merge__(self, band: str = None):
        """
        Alternative method to __mosaic__ a bit tricky because, by default it wants to load all file/s in RAM
        :param band:
        :return:
        """
        band_file = self[band]
        if os.path.exists(band_file):
            return

        files = self.files[band]
        band_file = os.path.join(self.root_folder, f'{band}.tif')

        mgrs_poly = self.s2_tiles[0].mgrs_geometry
        asset_name = SENTINEL2_BAND_MAP[band]
        asset_def = self.s2_tiles[0].assets[asset_name]
        no_data = asset_def['raster:bands'][0]['nodata']
        res = asset_def['raster:bands'][0]['spatial_resolution']

        profile = dict(
            tiled=True,
            compress='ZSTD',
            bigtiff="IF_SAFER",
            driver="GTiff",
            nodata=no_data
        )
        merge(sources=files,bounds=mgrs_poly.bounds, res=res,nodata=no_data,method='first',
              target_aligned_pixels=True,mem_limit=2.1,dst_path=band_file, dst_kwds=profile)
        self[band] = band_file





    def download(self, bands: List[str] = None, progress: Progress = None,
                       connect_timeout=25, read_timeout=900, force=False):
        return asyncio.run(
            self.__download__(bands=bands, progress=progress, connect_timeout=connect_timeout,
                          read_timeout=read_timeout, force=force)
        )