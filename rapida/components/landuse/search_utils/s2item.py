import asyncio
import datetime
import shutil
import threading
from rasterio.shutil import copy as rio_copy
from rasterio.windows import bounds, from_bounds
from rasterio.warp import calculate_default_transform, aligned_target
from osgeo import gdal
from pyproj import CRS
import os.path
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rich.progress import Progress
import aiohttp
import aiofiles
import random
from contextlib import suppress
from rapida.components.landuse.constants import SENTINEL2_ASSET_MAP, SENTINEL2_BAND_MAP
from rapida.components.landuse.search_utils.mgrstiles import Candidate
from rapida.components.landuse.search_utils.mgrsconv import _parse_mgrs_100k
from typing import Callable, Optional, Union



from typing import List, Tuple
import math
import logging
gdal.UseExceptions()
logger = logging.getLogger(__name__)
rlog = logging.getLogger('rasterio')
rlog.setLevel(logging.ERROR)
RasterValidator = Union[bool, Callable[[str], bool]]


def _is_valid_raster(path: str) -> bool:
    try:
        # "Valid" here = rasterio can open it successfully.
        with rasterio.open(path) as src:
            # Optional: tighten validity if you want
            # return bool(src.count and src.width and src.height and src.crs)
            return True
    except Exception:
        return False


async def _cleanup_invalid_file(src):
    if os.path.exists(src) and not _is_valid_raster(src):
        with suppress(FileNotFoundError):
            os.remove(src)


def compute_offsets(remote_size: int, chunks: int) -> List[Tuple[int, int]]:
    """
    Split [0, remote_size) into <= nchunks contiguous, gap-free, non-overlapping ranges.
    Final chunk absorbs the remainder.
    """
    if chunks <= 0:
        raise ValueError("nchunks must be >= 1")
    if remote_size <= 0:
        return [(0, 0)]

    # ceil() ensures we never create more than `nchunks` tasks
    chunk_size = math.ceil(remote_size / chunks)
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

    def __init__(self, mgrs_grid:str=None, s2_tiles:List[Candidate] = None,
                 target_resolution=10, target_crs=None,
                 workdir:str = None, concurrent_bands=50):

        self.utm_zone, self.mgrs_band, self.mgrs_grid_letters = _parse_mgrs_100k(grid_id=mgrs_grid)
        self.mgrs_grid = mgrs_grid
        self.s2_tiles = sorted(s2_tiles, key=lambda c:-c.quality_score)
        self.workdir = os.path.join(os.path.abspath(workdir), self.mgrs_grid)
        self.target_crs = CRS.from_user_input(target_crs)
        self.target_resolution=target_resolution or None
        self.__urls__ = {}
        self.__files__ = {}
        self.__bands__ = {}
        self.__assets__ = {}
        self.__resolutions__ = {}
        self._task = None
        self._loop = None
        self._prepare_urls_()
        os.makedirs(self.workdir, exist_ok=True)
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
                    if not asset_band in self.__resolutions__:
                        self.__resolutions__[asset_band] = cand.assets[asset_name]['raster:bands'][0]['spatial_resolution']
                if not asset_band in self.__assets__:
                    self.__assets__[asset_band] = asset_name

    @property
    def urls(self):
        return self.__urls__

    @property
    def files(self):
        return self.__files__

    @property
    def bands(self):
        return sorted(self.__assets__.keys())

    @property
    def assets(self):
        return tuple(sorted(self.__assets__.values()))


    @property
    def resolution(self):
        return min(self.__resolutions__.values())

    @property
    def vrts(self):
        vrts = {}
        for band in self.bands:
            vrts[band] = self.__vrt__(band=band)
        return vrts

    def __vrt__(self, band=None):
        """
        Create a VRT in target_crs and target resolution with same size as the ideal
        MGRS grid
        :param band:
        :return:
        """
        if not band in self.bands:
            raise ValueError(f'Sentinel2 band "{band}" is invalid. Valid bands are {"".join(self.bands)}')

        band_file = self[band]

        # compute mgrs ideal height and width from mgrs_geometry of teh first candidate
        mgrs_poly = self.s2_tiles[0].mgrs_geometry
        left, bottom,right, top = mgrs_poly.bounds
        yres, xres = self.target_resolution, self.target_resolution
        mgrs_height = math.floor((top-bottom)/yres)
        mgrs_width = math.floor((right-left)/yres)

        if band_file is not None:
            vrt_path = band_file.replace('.tif', '.vrt')
            with rasterio.open(band_file) as src:


                # # compute the default transform after reprojection, add 2  extra pixels for some overlap
                transform, twidth, theight = calculate_default_transform(
                    src.crs, self.target_crs, mgrs_width+2, mgrs_height+2, *mgrs_poly.bounds,resolution=self.target_resolution
                )
                #
                #align nicely the transform to pixel, the extra pixels (2) ensure the image is large enough
                aligned_transform, w, h = aligned_target(transform=transform, width=twidth, height=theight,
                                          resolution=self.target_resolution)
                # use nicely aligned transform and computed height and width
                with WarpedVRT(src, crs=self.target_crs, dtype=src.dtypes[0], nodata=src.nodata, transform=aligned_transform,
                               width=w, height=h,resampling=Resampling.nearest, count=src.count) as vrt:

                    rio_copy(vrt, vrt_path, driver="VRT")

                    return vrt_path


    def __setitem__(self, band, band_file):
        assert band in self.bands, f'Sentinel2 band "{band}" is invalid. Valid bands are {"".join(self.bands)}'
        self.__bands__[band] = band_file

    def __getitem__(self, band):
        if not band in self.bands:
            raise ValueError(f'Sentinel2 band "{band}" is invalid. Valid bands are {"".join(self.bands)}')
        band_file = self.__bands__.get(band, None)
        if band_file is None:
            logger.debug(f'There is no imagery for MGRS {self.mgrs_grid} and band {band} in {self.workdir}.Consider downloading it first ')
        return band_file

    def _is_valid_raster(self, path: str) -> bool:
        try:
            with rasterio.open(path):
                return True
        except Exception as e:
            logger.error(e)
            return False

    async def __fetch_range__(
            self, session: aiohttp.ClientSession, url: str, start: int, end: int,
            max_retries: int = 5, base_backoff: float = 5
    ) -> tuple[bytes, int]:
        headers = {"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"}
        expected = end - start
        attempt = 0
        while True:
            try:
                async with session.get(url, headers=headers) as resp:
                    # Accept 206; handle 200 explicitly (server ignored Range)
                    if resp.status == 206:
                        data = await resp.read()
                    elif resp.status == 200:
                        data = await resp.read()
                        # Entire body came back; treat as mismatch for this slice
                        if len(data) != expected:
                            raise IOError(
                                f"Server returned 200 to Range {start}-{end - 1}; "
                                f"expected {expected}B slice, got full {len(data)}B"
                            )
                    else:
                        resp.raise_for_status()

                    if len(data) != expected:
                        raise IOError(f"Expected {expected}B, got {len(data)}B for {start}-{end}")
                    return data, start

            except aiohttp.ClientResponseError as e:
                attempt += 1
                # Retry 5xx and 408/429 only
                retryable = (500 <= e.status < 600) or e.status in (408, 429)
                if not retryable or attempt > max_retries:
                    raise
                delay = base_backoff * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                logger.warning(
                    f"Range {start}-{end} {e.status} {e.message}; retry {attempt}/{max_retries} in {delay:.2f}s")
                await asyncio.sleep(delay)

            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError,
                    aiohttp.ClientPayloadError, asyncio.TimeoutError) as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                delay = base_backoff * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                logger.warning(f"Range {start}-{end} {type(e).__name__}; retry {attempt}/{max_retries} in {delay:.2f}s")
                await asyncio.sleep(delay)
            except Exception as e:
                logger.warning(f"failure Range {start}-{end} {type(e).__name__}; retry {attempt}/{max_retries} in {delay:.2f}s")

                raise e


    async def __download_file__(
            self,
            session: aiohttp.ClientSession = None,
            url: str = None,
            dst: str = None,
            chunks: int = 1,
            timeout_minutes=30,
            force=False
    ):

        async def _cancel_all(tsks):
            for t in tsks:
                if not t.cancelled():
                    t.cancel()
            await asyncio.gather(*tsks, return_exceptions=True)

        async def _cleanup_tmp(path: str):
            with suppress(FileNotFoundError):
                os.remove(path)
        # Cap ranges to limit peak memory
        chunks = max(1, min(int(chunks), 32))
        async with self.semaphore:
            dstdir = os.path.dirname(dst) or "."
            os.makedirs(dstdir, exist_ok=True)
            tmp = dst + ".part"

            # 1) HEAD: discover size
            async with session.head(url) as head_resp:
                head_resp.raise_for_status()
                cl = head_resp.headers.get("content-length")
                if cl is None:
                    raise ValueError("No content-length in response headers")
                remote_content_length = int(cl)

            # 2) Short-circuit if already valid (unless force)
            if not force and os.path.exists(dst):
                if self._is_valid_raster(dst):
                    logger.debug(f'Reusing {dst}')
                    return dst
                with suppress(FileNotFoundError):
                    logger.info(f'Re-downloading {dst}')
                    os.remove(dst)

            # ensure no stale .part
            with suppress(FileNotFoundError):
                os.remove(tmp)

            # 3) Launch chunk tasks
            offsets = compute_offsets(remote_content_length, chunks)
            tasks = [
                asyncio.create_task(
                    self.__fetch_range__(session=session, url=url, start=start_offset, end=end_offset),
                    name=f"{start_offset}-{end_offset}",
                )
                for (start_offset, end_offset) in offsets
            ]

            try:
                # 4) Write chunks to TMP (global timeout)
                async with aiofiles.open(tmp, "wb") as f:
                    async with asyncio.timeout(timeout_minutes * 60):
                        for fut in asyncio.as_completed(tasks):
                            data, offset = await fut  # may raise
                            await f.seek(offset)
                            await f.write(data)


                # 5) Validate size
                if not os.path.exists(tmp):
                    raise IOError(f"Downloaded temporary file {tmp} does not exist!")
                size = os.path.getsize(tmp)
                if size != remote_content_length:
                    with suppress(FileNotFoundError):
                        os.remove(tmp)
                    raise IOError(f"Downloaded temporary file {tmp} is incomplete: expected size {remote_content_length}, got {size}")

                # 6) Atomically publish
                os.replace(tmp, dst)

                # 6) Optional raster validation
                if not self._is_valid_raster(dst):
                    with suppress(FileNotFoundError):
                        os.remove(dst)
                    raise IOError(f"Downloaded temporary  file {dst} is invalid")
                return dst

            except asyncio.CancelledError:
                with suppress(Exception):
                    await _cancel_all(tasks)
                    await asyncio.shield(_cleanup_tmp(tmp))
                raise
            except Exception:
                with suppress(Exception):
                    await _cancel_all(tasks)
                    await asyncio.shield(_cleanup_tmp(tmp))
                raise
            finally:
                # Ensure no stray tasks/temp
                pend = [t for t in tasks if not t.done()]
                if pend:
                    with suppress(Exception):
                        await _cancel_all(pend)
                with suppress(Exception):
                    await asyncio.shield(_cleanup_tmp(tmp))



    async def __download_band__(self, band: str = None, session=None,
                                progress=None, progress_task=None, force=False, mosaic=True,
                                stop: Optional[threading.Event] = None):
        """
        Download a S2 image band from one or more files
        :param band:
        :param session:
        :param progress:
        :param progress_task:
        :param force:
        :param mosaic:
        :param stop:
        :return:
        """
        urls = self.urls[band]
        tasks = []
        downloaded = []
        indices = {}
        band_folder = os.path.join(self.workdir, band)
        for i, url in enumerate(urls):
            chunk = url.split(self.mgrs_grid_letters)[-1]
            year, month, day, *rest = chunk[1:].split('/')
            band, _  = rest[-1].split('.')
            date = datetime.date(year=int(year), month=int(month), day=int(day))
            original_file_name = '_'.join(rest)
            new_file_name = f"{date.strftime('%Y%m%d')}_{original_file_name}"
            local_file_path = os.path.join(band_folder, new_file_name)
            indices[new_file_name] = i
            task = asyncio.create_task(
                        self.__download_file__(session=session, url=url, dst=local_file_path, force=force),
                        name=new_file_name
                    )
            tasks.append(task)
        try:
            for done in asyncio.as_completed(tasks):  # great for progress updates
                # cooperative stop between awaits
                if stop is not None and stop.is_set():
                    raise asyncio.CancelledError()
                downloaded_file = await done  # will raise if that task failed
                _, file_name = os.path.split(downloaded_file)
                index=indices[file_name]
                downloaded.append((downloaded_file, index))  # preserve input order
                if progress is not None and progress_task is not None:
                    # stable description; only advance the counter
                    progress.update(progress_task, advance=1)

            downloaded = [e[0] for e in sorted(downloaded, key=lambda el: el[1])]
            self.__files__[band] = tuple(downloaded)
            if progress is not None and progress_task is not None:
                progress.update(
                                progress_task,
                                description=f'[red]Downloaded band {band} in MGRS-{self.mgrs_grid}'
                            )
            if mosaic:
                return self.__merge__(band=band, progress=progress, force=force)
            return downloaded
        except Exception as e:
            # cancel anything still running
            logger.info(f'Cancelling all tasks for {band} in MGRS {self.mgrs_grid}: {e}')
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            shutil.rmtree(band_folder)
            raise



    async def __download__(self, bands: List[str] = None, stop: threading.Event = None, progress: Progress = None,
                           connect_timeout=25, read_timeout=900, force=False):

        # Accurate total by summing actual URLs per band.
        nfiles = sum(len(self.urls[b]) for b in bands)

        timeout = aiohttp.ClientTimeout(total=None, connect=connect_timeout, sock_read=read_timeout)
        connector = aiohttp.TCPConnector(enable_cleanup_closed=True, ttl_dns_cache=300)
        progress_task = None

        downloaded_bands: dict[str, object] = {}  # <-- successes only
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            band_tasks: set[asyncio.Task] = set()

            if progress is not None:
                progress_task = progress.add_task(
                    description=f"[red]Downloading {len(bands)} Sentinel-2 bands in MGRS grid {self.mgrs_grid}",
                    total=nfiles
                )
            # schedule all bands
            for band in bands:
                assert band in self.bands, (
                    f'Sentinel2 band "{band}" is invalid. Valid bands are {", ".join(self.bands)}'
                )
                if stop is not None and stop.is_set():
                    raise asyncio.CancelledError()

                t = asyncio.create_task(
                    self.__download_band__(band=band, session=session,
                                           progress=progress, progress_task=progress_task,
                                           force=force, stop=stop),
                    name=band
                )
                band_tasks.add(t)
            while band_tasks:
                if stop is not None and stop.is_set():
                    raise asyncio.CancelledError()

                done, pending = await asyncio.wait(band_tasks, return_when=asyncio.FIRST_COMPLETED)

                # extra cooperative stop check
                if stop is not None and stop.is_set():
                    raise asyncio.CancelledError()

                for dt in done:
                    band = dt.get_name()
                    try:
                        res = await dt  # merge path or None if you choose
                        downloaded_bands[band] = res  # <-- collect ONLY good results
                    except asyncio.CancelledError:
                        # propagate user cancel
                        raise
                    except Exception as e:
                        # ignore failures by design; lower layers already cleaned partials
                        logger.error(f'Failed to download band {band} in MGRS grid [{self.mgrs_grid}]: {e}')



                band_tasks = pending

            if progress is not None and progress_task is not None:
                progress.update(
                    progress_task,
                    description=f"[red]Downloaded {len(downloaded_bands)} Sentinel-2 bands in MGRS grid {self.mgrs_grid}"
                )

        return downloaded_bands



    def __merge__(self, band: str = None, stop=None, progress=None, force=None):
        band_file = os.path.join(self.workdir, f'{band}.tif')
        if not force:
            try:
                with rasterio.open(band_file) as src:
                    if src.crs and src.width and src.height:
                        self[band] = band_file
                        return band_file
            except Exception:
                if os.path.exists(band_file):
                    logger.info(f'{band_file} will be recreated!')
                    os.remove(band_file)

        files = self.files[band]

        asset_name = SENTINEL2_BAND_MAP[band]
        asset_def = self.s2_tiles[0].assets[asset_name]
        no_data = asset_def['raster:bands'][0]['nodata']

        best = files[0]
        src_datasets, vrt_datasets = [], []
        progress_task = None

        try:
            with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=1024, OPJ_NUM_THREADS="ALL_CPUS",
                              USE_TILE_AS_BLOCK="YES"):
                with rasterio.open(best) as src:
                    mgrs_profile = src.profile
                    mgrs_profile.update(dict(tiled=True, compress='ZSTD', bigtiff="IF_SAFER",
                                             driver="GTiff", nodata=no_data))
                    vrt_opts = dict(crs=src.crs, transform=src.transform, width=src.width,
                                    height=src.height, resampling=Resampling.nearest)

                    src_datasets = [src]
                    vrt_datasets = []
                    for p in files[1:]:
                        s = rasterio.open(p)
                        src_datasets.append(s)
                        vrt_datasets.append(WarpedVRT(s, **vrt_opts))

                    with rasterio.open(band_file, "w", **mgrs_profile) as dst:
                        wins = {block: win for block, win in dst.block_windows()}
                        if progress is not None:
                            w = 'image' + ('s' if len(files) > 1 else '')
                            progress_task = progress.add_task(
                                description=f"[red]Merging {len(files)} Sentinel2 {w} for band {band} in {self.mgrs_grid}",
                                total=len(wins)
                            )

                        for block, window in wins.items():
                            if stop is not None and stop.is_set():
                                raise KeyboardInterrupt
                            bbox = bounds(window=window, transform=dst.transform)
                            src_win = from_bounds(*bbox, transform=src.transform).round()
                            out = src.read(1, window=src_win)
                            holes = (out == no_data)

                            if not holes.any():
                                dst.write(out, 1, window=window)
                                if progress and progress_task:
                                    progress.update(progress_task, advance=1)
                                continue

                            for v in vrt_datasets:
                                if not holes.any():
                                    break
                                arr = v.read(1, window=src_win)
                                valid = (arr != no_data)
                                take = holes & valid
                                if take.any():
                                    out[take] = arr[take]
                                    holes[take] = False

                            if holes.any():
                                logger.warning(
                                    f'Some pixels are still empty in {self.mgrs_grid}-{band}-{(window.row_off, window.col_off, window.height, window.width)}')
                                out[holes] = no_data

                            dst.write(out, 1, window=window)
                            if progress and progress_task:
                                progress.update(progress_task, advance=1)

            self[band] = band_file
            return band_file
        except KeyboardInterrupt:
            if not self._is_valid_raster(band_file):
                with suppress(FileNotFoundError):
                    os.remove(band_file)
            raise
        finally:
            for vrt in vrt_datasets: vrt.close()
            for s in src_datasets[1:]: s.close()
            if progress is not None and progress_task is not None:
                progress.remove_task(progress_task)
            # if progress and progress_task:
            #     try:
            #         progress.stop_task(progress_task)
            #     except Exception:
            #         pass

    def download(self, bands=None, stop: threading.Event = None, progress=None,
                 connect_timeout=25, read_timeout=900, force=False):

        if threading.current_thread() is threading.main_thread():
            # if you ever call it in main thread
            return asyncio.run(self.__download__(
                bands=bands, stop=stop, progress=progress,
                connect_timeout=connect_timeout, read_timeout=read_timeout, force=force
            ))

        # worker thread branch
        loop = asyncio.new_event_loop()
        self._loop = loop  # keep a handle so main can cancel
        asyncio.set_event_loop(loop)
        try:
            self._task = loop.create_task(self.__download__(
                bands=bands, stop=stop, progress=progress,
                connect_timeout=connect_timeout, read_timeout=read_timeout, force=force
            ), name=f'down{self.mgrs_grid}')

            # run until task completes or is cancelled
            return loop.run_until_complete(self._task)

        except asyncio.CancelledError:
            # task was cancelled from main
            logger.debug(f"Worker {threading.current_thread().name}  downloading {self.mgrs_grid} was cancelled!")
            try:
                loop.run_until_complete(asyncio.gather(self._task, return_exceptions=True))
            except Exception:
                pass
            raise
        finally:
            # 1) cancel & drain any remaining tasks in this loop
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            except Exception:
                pending = []
            for t in pending: t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            # 2) shutdown async generators
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass

            asyncio.set_event_loop(None)
            loop.close()
            self._loop = None
            self._task = None


    # def merge(self, band: str = None):
    #     """
    #     Alternative method to __mosaic__ a bit tricky because, by default it wants to load all file/s in RAM
    #     :param band:
    #     :return:
    #     """
    #     band_file = self[band]
    #     if os.path.exists(band_file):
    #         return
    #
    #     files = self.files[band]
    #     band_file = os.path.join(self.workdir, f'{band}.tif')
    #
    #     mgrs_poly = self.s2_tiles[0].mgrs_geometry
    #     asset_name = SENTINEL2_BAND_MAP[band]
    #     asset_def = self.s2_tiles[0].assets[asset_name]
    #     no_data = asset_def['raster:bands'][0]['nodata']
    #     res = asset_def['raster:bands'][0]['spatial_resolution']
    #
    #     profile = dict(
    #         tiled=True,
    #         compress='ZSTD',
    #         bigtiff="IF_SAFER",
    #         driver="GTiff",
    #         nodata=no_data
    #     )
    #     merge(sources=files,bounds=mgrs_poly.bounds, res=res,nodata=no_data,method='first',
    #           target_aligned_pixels=True,mem_limit=2.1,dst_path=band_file, dst_kwds=profile)
    #     self[band] = band_file


