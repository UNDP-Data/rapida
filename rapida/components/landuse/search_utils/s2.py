import asyncio
import datetime
import os.path
from rich.progress import Progress
import aiohttp
import aiofiles
import random
from rapida.components.landuse.constants import SENTINEL2_ASSET_MAP
from rapida.components.landuse.search_utils.tiles import Candidate
from rapida.components.landuse.search_utils.zones import _parse_mgrs_100k
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

    def __init__(self, mgrs_grid:str=None, s2_tiles:List[Candidate] = None, root_folder:str = None):

        self.utm_zone, self.mgrs_band, self.mgrs_grid_letters = _parse_mgrs_100k(grid_id=mgrs_grid)
        self.mgrs_grid = mgrs_grid
        self.s2_tiles = sorted(s2_tiles, key=lambda c:-c.quality_score)
        self.root_folder = os.path.join(os.path.abspath(root_folder), self.mgrs_grid)
        os.makedirs(self.root_folder, exist_ok=True)
        self.__files__ = {}
        self._prepare_file_links_()

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


    def _prepare_file_links_(self):
        if not self.__files__:
            for asset_name, asset_band in SENTINEL2_ASSET_MAP.items():
                for cand in self.s2_tiles:
                    asset_s3_uri = cand.assets[asset_name]['href']
                    asset_html_uri = self._s3_to_http(asset_s3_uri)
                    if asset_band in self.__files__:
                        self.__files__[asset_band].append(asset_html_uri)
                    else:
                        self.__files__[asset_band] = [asset_html_uri]
    @property
    def files(self):
        return self.__files__

    @property
    def bands(self):
        return sorted(self.__files__.keys())

    def __getitem__(self, key):
        assert key in self.bands, f'Sentinel2 band "{key}" is invalid. Valid bands are {"".join(self.bands)}'
        logger.info('GOTCHA')

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
        # 1) Discover size (HEAD)
        async with session.head(url) as head_resp:
            head_resp.raise_for_status()
            cl = head_resp.headers.get("content-length")
            if cl is None:
                raise ValueError("No content-length in response headers")
            remote_content_length = int(cl)

        offsets = compute_offsets(remote_content_length, nchunks)

        # 2) Launch chunk range tasks
        tasks: list[asyncio.Task] = []
        for start_offset, end_offset in offsets:
            t = asyncio.create_task(
                self.__fetch_range__(session=session, url=url,
                                     start=start_offset, end=end_offset),
                name=f"{start_offset}-{end_offset}"
            )
            tasks.append(t)

        # 2.5) early return if file exists, seems to be of identical size and force=False
        if os.path.exists(dst):
            if os.path.getsize(dst) == remote_content_length and not force:
                return dst

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

    async def __download_band__(self, band:str=None, session=None, progress=None, progress_task=None, force=False):
        """
        Download files that will be sued to make a single band mosaic
        :param band:
        :param session:
        :param progress:
        :param progress_task:
        :return:
        """
        band_urls = self.files[band]
        tasks = []
        for url in band_urls:
            chunk = url.split(self.mgrs_grid_letters)[-1]
            year, month, day, *rest = chunk[1:].split('/')
            date = datetime.datetime(year=int(year), month=int(month), day=int(day))
            original_file_name = rest[-1]
            new_file_name = f"{date.strftime('%Y%m%d')}_{original_file_name}"
            file_path = os.path.join(self.root_folder,band,new_file_name )

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

        return band


    async def download(self, bands: List[str] = None, progress: Progress = None,
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

        # Mildly tuned connector for parallel ranged GETs
        connector = aiohttp.TCPConnector(
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
            # You can set limits here if you like; leaving defaults is fine too.
            # limit_per_host=len(bands) or a small constant; omit if unsure.
        )

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            band_tasks: set[asyncio.Task] = set()

            progress_task = None
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
                        try:
                            await dt  # consume result or raise
                        except Exception as e:
                            band = dt.get_name()
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
