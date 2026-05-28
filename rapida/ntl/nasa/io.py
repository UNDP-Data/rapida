from pathlib import Path
import httpx
from rapida.ntl.nasa.util import get_intersecting_tiles
import asyncio
import os
from rich.progress import Progress
from rapida.ntl import cache
from rapida.util.download_remote_file import download_remote_files
import logging
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
    headers = {"Authorization": f"Bearer {ea_token}"}
    return await download_remote_files(
        file_urls=urls,dst_folder=dst_dir, progress=progress, headers=headers
    )
