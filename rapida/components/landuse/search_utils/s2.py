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

    async def __fetch_range__( self,
            session: aiohttp.ClientSession = None,
            url: str = None,
            start: int = None,
            end: int = None,
            max_retries: int = 4,
            base_backoff: float = 0.25,
    ) -> bytes:
        """
        Fetch [start, end) with HTTP Range and bounded retries (exponential backoff + jitter).
        """
        headers = {"Range": f"bytes={start}-{end - 1}"}
        attempt = 0
        while True:
            try:
                async with session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    content = await resp.read()
                    expected = end - start
                    if len(content) != expected:
                        raise IOError(f"Expected {expected}B, got {len(content)}B for {start}-{end}")
                    return content
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                delay = (base_backoff * (2 ** (attempt - 1))) * (1 + random.random() * 0.25)
                logger.warning(f"Range {start}-{end} failed ({e}); retry {attempt}/{max_retries} in {delay:.2f}s")
                await asyncio.sleep(delay)


    async def __download_file__(self, session:aiohttp.ClientSession=None,
                                url:str=None, dst:str = None, nchunks = 5):

        head_resp = await session.head(url)
        head_resp.raise_for_status()
        # fetch content-length from remote file header
        remote_content_length = head_resp.headers.get("content-length")
        if remote_content_length is None:
            raise ValueError("No content-length in response headers")
        remote_content_length = int(remote_content_length)
        offsets = compute_offsets(remote_content_length,nchunks)

        tasks = []
        for start_offset, end_offset in offsets:
            tasks.append(
                asyncio.create_task(
                    self.__fetch_range__(session=session, url=url,
                                            start=start_offset,
                                            end=end_offset),
                    name=f'{start_offset}-{end_offset}'
                )

            )
        async with aiofiles.open(dst, 'wb') as local_file:
            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for done_task in done:
                    soff,eoff = map(int, done_task.get_name().split('-'))
                    try:
                        bytes = await done_task  # already finished; get result or raise
                        await  local_file.seek(soff)
                        await local_file.write(bytes)
                        logger.debug(f'Successfully downloaded and saved {len(bytes)} bytes into {dst} ')

                    except Exception as e:
                        # yield t, e  # surface per-task error without killing others
                        logger.error(f'Failed to download bytes {soff}-{eoff} because {e}')
                        raise


        assert os.path.exists(dst), f'{dst} does not exist'
        assert os.path.getsize(dst) == remote_content_length, f'{dst} was NOT downloaded successfully'
        return dst
    async def __download_band__(self, band:str=None, session=None, progress=None, progress_task=None):
        band_urls = self.files[band]
        tasks = []
        for url in band_urls:
            chunk = url.split(self.mgrs_grid_letters)[-1]
            year, month, day, *rest = chunk[1:].split('/')

            date = datetime.datetime(year=int(year), month=int(month), day=int(day))
            original_file_name = rest[-1]
            new_file_name = f"{date.strftime('%Y%m%d')}_{original_file_name}"
            file_path = os.path.join(self.root_folder,new_file_name )

            tasks.append(
                asyncio.create_task(
                    self.__download_file__(session=session, url=url, dst=file_path),
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
                except Exception as e:
                    # yield t, e  # surface per-task error without killing others
                    import traceback
                    traceback.print_exc(e)
                    logger.error(f'Failed to download {band} because {e}')

        return band

    async def download(self, bands:List[str] = None, progress:Progress=None, connect_timeout=25, read_timeout=900):


        nfiles = len(self.s2_tiles) * len(bands)
        timeout = aiohttp.ClientTimeout(total=None, connect=connect_timeout, sock_connect=read_timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            band_tasks = set()
            progress_task = None
            if progress is not None:
                progress_task = progress.add_task(
                    description=f'[red]Going to download Sentinel2 data  in {len(band_tasks)} MGRS 100K grids ',
                    total=nfiles)
            try:
                for band in bands:
                    assert band in self.bands, f'Sentinel2 band "{band}" is invalid. Valid bands are {"".join(self.bands)}'
                    band_task = asyncio.create_task(
                        self.__download_band__(band=band, session=session,
                                               progress=progress, progress_task=progress_task),
                        name=band
                    )
                    band_tasks.add(band_task)

                while band_tasks:
                    done, band_tasks = await asyncio.wait(band_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for done_task in done:
                        band = done_task.get_name()
                        try:
                            result = await done_task  # already finished; get result or raise

                        except Exception as e:
                            #yield t, e  # surface per-task error without killing others
                            logger.error(f'Failed to download {band} because {e}')

            except asyncio.CancelledError as ce:
                logger.info(f'Cancelling download for {self.mgrs_grid}')
                for t in band_tasks:
                    t.cancel()
                await asyncio.gather(*band_tasks, return_exceptions=True)
                raise
            finally:
                for t in band_tasks:
                    t.cancel()
                await asyncio.gather(*band_tasks, return_exceptions=True)
                # if progress is not None and progress_task is not None:
                #     progress.remove_task(progress_task)
