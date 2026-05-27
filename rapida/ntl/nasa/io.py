from pathlib import Path
import httpx
from rapida.ntl.nasa.util import get_intersecting_tiles
import os
from rich.progress import Progress
from rapida.ntl import cache


async def download(timestamp: str = None, product: str = None, dst_dir:str=None, progress:Progress=None):


    key = f'{product.upper()}_{timestamp}'


    # EarthAccess token
    ea_token = os.environ.get('EARTHDATA_TOKEN')
    headers = {"Authorization": f"Bearer {ea_token}"}


    # tasks = []
    # semaphore = asyncio.Semaphore(5)
    # async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
    #     pass