from typing import Iterable
from osgeo.gdal import Info
import httpx
import os
import aiohttp
import aiofiles
import asyncio
import logging

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_CONCURRENCY = 5


async def download_remote_file(file_url: str,
                               output_file: str,
                               progress=None,)->str:
    """
    Download a file from a remote URL to a local file path.

    Note. file_url must be either HTTP or HTTPS. If URL starts with 'az:' or 'geohub:',
    it must be interpolated before using this function.

    :param file_url: The URL of the file to download.
    :param output_file: The local file path of the file to download.
    :param progress: An optional rich progress bar instance.
    :return: The local file path downloaded.
    """
    download_task = None

    try:
        if progress is not None:
            download_task = progress.add_task(
                description=f'[blue] Downloading {file_url}', total=None)

        async with httpx.AsyncClient() as client:
            head_resp = await client.head(file_url)
            head_resp.raise_for_status()
            # fetch content-length from remote file header
            remote_content_length = head_resp.headers.get("content-length")
            if remote_content_length is None:
                raise ValueError("No content-length in response headers")
            remote_content_length = int(remote_content_length)

            if os.path.exists(output_file):
                local_file_size = os.path.getsize(output_file)
                if local_file_size == remote_content_length:
                    logging.debug(f"File already exists and size matches. Skipped: {output_file}")
                    return output_file
                else:
                    logging.debug(f"File size mismatch. Removing local file: {output_file}")
                    os.remove(output_file)

            if not os.path.exists(output_file):
                async with client.stream("GET", file_url) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))

                    if progress is not None and download_task is not None:
                        progress.update(download_task, total=total)

                    with open(output_file, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            if progress is not None and download_task is not None:
                                progress.update(download_task, advance=len(chunk))

        return output_file
    finally:
        if progress and download_task:
            progress.update(download_task, description=f"[blue] Downloaded file saved to {output_file}")
            progress.remove_task(download_task)


async def fetch_and_write(session=None, url=None, offset=None, size=None, file_descriptor=None, sem=None, task_id=None, progress=None):
    try:
        headers = {'Range': f'bytes={offset}-{offset + size - 1}'}
        async with sem:
            resp = await session.get(url, headers=headers)
            resp.raise_for_status()
            content = await resp.read()
            await file_descriptor.seek(offset)
            await file_descriptor.write(content)
            if progress and task_id is not None:
                progress.update(task_id, advance=len(content))

    except Exception as e:
        logger.error(f"Failed to fetch and write {url} to {file_descriptor.name}: {e}")
        raise


async def download_file(file_url=None, dst_file_path=None,
                        session=None, force=False,
                        chunk_size=CHUNK_SIZE, no_attempts=3,
                        data_read_timeout=None,
                        progress=None
                        ):
    for attempt in range(no_attempts):
        try:
            down_task = None
            async with session.get(file_url, timeout=data_read_timeout) as response:
                if response.status == 200:

                    remote_size = int(response.headers['Content-Length'])
                    if os.path.exists(dst_file_path):
                        if not force and os.path.getsize(dst_file_path) == remote_size:
                            logger.debug(f'Returning local file {dst_file_path}')
                            return dst_file_path
                        else:
                            os.remove(dst_file_path)

                    if progress:
                        down_task = progress.add_task(f'[cyan]Downloading {file_url}', total=remote_size)
                    async with aiofiles.open(dst_file_path, 'wb') as local_file:
                        while True:
                            chunk = await response.content.read(chunk_size)
                            if not chunk:
                                break
                            await local_file.write(chunk)
                            if progress and down_task is not None:
                                progress.update(down_task, advance=len(chunk))

                    size = os.path.getsize(dst_file_path)
                    if size != remote_size:
                        raise Exception(f'{file_url} is was not downloaded correctly!')

                    logger.debug(f'File {dst_file_path} was successfully downloaded')

                    return dst_file_path
                else:
                    msg = f'GET request failed for url {file_url} with status code {response.status}.'
                    raise Exception(msg)
        except (KeyboardInterrupt, asyncio.CancelledError) as ce:
            logger.error(
                f'Download action for {file_url} was cancelled by the user')
            if os.path.exists(dst_file_path):
                os.remove(dst_file_path)
            raise ce
        except Exception as e:
            logger.error(f'Exception "{e}" was encountered in while downloading {file_url}')
            if os.path.exists(dst_file_path):
                os.remove(dst_file_path)
            if attempt == no_attempts - 1:
                raise e
            continue
        finally:
            if down_task and progress:
                progress.remove_task(down_task)



async def download_s3_object(
    url,
    filename,
    chunk_size: int = CHUNK_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress=None,
    max_retries: int = 5,
    client_session=None,
):
    attempt = 0
    while attempt < max_retries:
        try:
            resp = await client_session.head(url)
            resp.raise_for_status()
            total_size = int(resp.headers['Content-Length'])

            async with aiofiles.open(filename, 'wb') as f:
                await f.truncate(total_size)

            task_id = None
            if progress:
                task_id = progress.add_task(f'[cyan]Downloading {url}', total=total_size)

            sem = asyncio.Semaphore(concurrency)
            tasks = []
            async with aiofiles.open(filename, 'r+b') as fd:
                for offset in range(0, total_size, chunk_size):
                    size = min(chunk_size, total_size - offset)

                    tasks.append(
                        asyncio.create_task(fetch_and_write(
                            session=client_session,
                            url=url,
                            offset=offset,
                            size=size,
                            file_descriptor=fd,
                            sem=sem,
                            task_id=task_id,
                            progress=progress
                        ))
                    )

                try:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                    for task in done:
                        if task.exception():
                            raise task.exception()
                    if progress and task_id is not None:
                        progress.remove_task(task_id)

                    logger.debug(
                        f"Downloaded {os.path.getsize(filename)} bytes from {url} to {filename}"
                    )
                    return filename
                finally:
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

        except asyncio.CancelledError:
            if os.path.exists(filename):
                os.remove(filename)
            raise
        except Exception as e:
            attempt += 1
            logger.warning(f"[Attempt {attempt}/{max_retries}] Failed to download {url}: {e}")
            if os.path.exists(filename):
                os.remove(filename)
            if attempt >= max_retries:
                logger.error(f"Exceeded max retries for {url}")
                raise
            await asyncio.sleep(2 ** attempt)
    return None


async def download_remote_files(
        file_urls: Iterable[str],
        dst_folder: str,
        progress=None,
        target_path_func=None,
        force=False, connect_timeout=250, data_read_timeout=9000
):
    """
    Download remote files from a list of URLs.

    :param file_urls: The URLs of the files to download.
    :param dst_folder: The folder to save the files in.
    :param progress: An optional rich progress bar instance.
    :param target_path_func: A function that takes a URL as an argument and returns a path to save the file to.
    """
    try:
        timeout = aiohttp.ClientTimeout(connect=connect_timeout, sock_connect=data_read_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as client_session:
            tasks = []
            os.makedirs(dst_folder, exist_ok=True)
            for file_url in file_urls:
                if target_path_func is not None:
                    target_path = target_path_func(file_url, dst_folder)
                    new_dirname = os.path.dirname(target_path)
                    if new_dirname != dst_folder:
                        os.makedirs(new_dirname, exist_ok=True)
                    file_name = os.path.basename(target_path)
                else:
                    file_name = os.path.basename(file_url)
                    target_path = os.path.join(dst_folder, file_name)

                tasks.append(
                    asyncio.create_task(
                        # download_s3_object(
                        #                 url=file_url,
                        #                 filename=target_path,
                        #                 progress=progress,
                        #                 client_session=client_session)
                        download_file(file_url=file_url,dst_file_path=target_path,
                                      session=client_session,force=force,
                                      progress=progress, data_read_timeout=data_read_timeout)

                    )


                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    # raise the error as if download on even one fails, the results should not be collected
                    raise result
                Info(result)
            return results
    except (KeyboardInterrupt, asyncio.CancelledError) as e:
        raise
    except Exception as e:
        logger.error(f"Failed to download remote files: {e}")
        raise
