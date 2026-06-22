import shelve
import time
import os
import tempfile
import hashlib
import json
from filelock import FileLock  # Added

MAX_AGE_SECONDS = 6 * 3600  # 6 hours
CACHE_PATH = os.path.join(tempfile.gettempdir(), "ntl_cache")
LOCK_PATH = f"{CACHE_PATH}.lock"  # Added


def search_id(search_params: dict) -> str:
    """Generates a deterministic unique ID based on the STAC search parameters."""
    param_string = json.dumps(search_params, sort_keys=True)
    return hashlib.md5(param_string.encode('utf-8')).hexdigest()


def store(key:str=None, value:str=None, tile:str=None, cache_path=CACHE_PATH):
    # Lock ensures only one process interacts with the file at a time
    with FileLock(LOCK_PATH, timeout=60):
        with shelve.open(cache_path) as cache:
            record = cache.get(key, None)
            if record is None:
                if tile:
                    record = {tile:value}, time.time()
                else:
                    record = value, time.time()
            else:
                tiles, creation_time = record
                if tile:
                    if not tile in tiles:
                        tiles.update({tile: value})
                    record = tiles, creation_time
                else:
                    record = value, creation_time
            cache[key] = record


def fetch(key:str=None, tile:str=None, cache_path=CACHE_PATH):
    # Lock ensures only one process interacts with the file at a time
    with FileLock(LOCK_PATH, timeout=60):
        with shelve.open(cache_path) as cache:
            record = cache.get(key, None)
            if record is None:
                return
            # 1. Directly unpack the tuple
            tiles, creation_time = record
            # 2. Check for expiration
            if time.time() - creation_time > MAX_AGE_SECONDS:
                del cache[key]
                return  # Expired
            # 3. Handle the tile request
            if tile:
                if tile in tiles:
                    return tiles[tile]
                return
            else:
                return tiles
if __name__ == '__main__':
    key = 'VJ146A3_202604'
    r = fetch(key=key, tile='h21v04')
    print(r)
    key = 'JRR-CloudMask_v3r2_j01_s202605140112102_e202605140113330_c202605140211319.nc'
    r = fetch(key=key)
    print(r)
    # ky = '32445566'
    #
    # store(key=ky, value='a')
    # r = fetch(key=ky)
    # print(r)
    # store(key=ky, value='b')
    # r = fetch(key=ky)
    # print(r)