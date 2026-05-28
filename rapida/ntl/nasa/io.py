from pathlib import Path
import httpx
from rapida.ntl.nasa.util import get_intersecting_tiles
import asyncio
import os
from rich.progress import Progress
from rapida.ntl import cache
from rapida.util.download_remote_file import download_remote_files
import logging
from urllib.parse import urlparse
from rapida.ntl.nasa.search import stac_search
from datetime import datetime
import numbers

logger = logging.getLogger(__name__)

async def download(timestamp: str = None, product: str = None, tile:str=None, dst_dir:str=None, progress:Progress=None):


    key = f'{product.upper()}_{timestamp}'
    urls = cache.fetch(key=key, tile=tile)


    if not urls:
        logger.info(f'Failed to locate information in {cache.CACHE_PATH} for {product}-{timestamp}-{tile or ""} \n' \
                       f'Consider searching first.')
        return 

    # EarthAccess token
    ea_token = os.environ.get('EARTHDATA_TOKEN')

    # Add this sanity check
    if not ea_token:
        raise ValueError("CRITICAL: EARTHDATA_TOKEN environment variable is not set or is empty!")

    headers = {"Authorization": f"Bearer {ea_token}"}
    return await download_remote_files(
        file_urls=urls,dst_folder=dst_dir, progress=progress, headers=headers
    )


async def download_tile(
        client: httpx.AsyncClient=None,
        url: str=None,
        dest_path: Path=None,
        semaphore: asyncio.Semaphore=None,
        progress: Progress=None,
        max_retries: int = 3
) -> Path:
    """
    Asynchronously downloads a single tile with exponential backoff retries.
    Uses a semaphore to prevent overwhelming the LANCE servers.
    """
    progress_task = None
    async with semaphore:
        for attempt in range(max_retries):
            try:
                # Start the streaming request
                async with client.stream("GET", url) as response:
                    response.raise_for_status()

                    # Initialize progress bar for this specific file
                    total_bytes = int(response.headers.get("Content-Length", 0))
                    if progress:
                        progress_task = progress.add_task(f'Downloading {url}', total=total_bytes,)
                    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
                    with open(tmp_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            if progress and progress_task:
                                progress.advance(progress_task, advance=len(chunk))
                    tmp_path.rename(dest_path)
                if progress and progress_task:
                    progress.update(progress_task, description=f"[green]✓ {dest_path.name}")

                return dest_path

            except httpx.HTTPError as e:
                if attempt == max_retries - 1:
                    if progress and progress_task:
                        progress.update(progress_task, description=f"[red]✗ {dest_path.name} (Failed)[/red]")
                    #progress.console.print(f"[red]Error downloading {url}: {e}[/red]")
                    return dest_path

                # Exponential backoff before retry (1s, 2s, 4s...)
                await asyncio.sleep(2 ** attempt)

            finally:
                if progress and progress_task:
                    progress.remove_task(progress_task)

async def bulk_download(bbox:tuple[numbers.Number]=None, start_date:datetime=None, end_date:datetime=None,
                           stream:str = None, products:str=None, dst_dir:str=None, progress=None):


    results = stac_search(stream=stream, products=products,
                       dt=[start_date, end_date], bbox=bbox,push_to_cache=False)


    # EarthAccess token
    ea_token = os.environ.get('EARTHDATA_TOKEN')

    # Add this sanity check
    if not ea_token:
        raise ValueError("CRITICAL: EARTHDATA_TOKEN environment variable is not set or is empty!")

    headers = {"Authorization": f"Bearer {ea_token}"}
    tasks = []
    semaphore = asyncio.Semaphore(5)
    progress_task = None
    if results:
        dest_path = Path(dst_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            for e in results:
                *r, url = e
                path = urlparse(url).path

                # 2. Get the basename from that path
                file_name = os.path.basename(path)

                filepath = dest_path / file_name

                tasks.append(asyncio.Task(
                    download_tile(client, url, filepath, semaphore), name=file_name
                ))

            if progress:
                progress_task = progress.add_task(description=f'Downloading {len(tasks)} images...', total=len(tasks))
            downloaded_files = []
            for task in asyncio.as_completed(tasks, timeout=20*len(tasks)):
                try:
                    downloaded_file = await task

                    if progress and progress_task is not None:
                        progress.update(progress_task,description=f'[green]🡇 {downloaded_file.name}', advance=1)
                    downloaded_files.append(str(downloaded_file))
                except Exception as e:
                    logger.error(e)

                except asyncio.CancelledError as ce:
                    for atask in tasks:
                        if not atask.done():
                            atask.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise