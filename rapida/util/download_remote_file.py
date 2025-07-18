from typing import Iterable

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


def download_remote_files(file_urls: Iterable[str],
                          dst_folder: str,
                          progress=None,
                          max_workers: int = 10,
                          max_retries:int = 5,
                          target_path_func=None):
    """
    Download remote files from a list of URLs.

    :param file_urls: The URLs of the files to download.
    :param dst_folder: The folder to save the files in.
    :param progress: An optional rich progress bar instance.
    :param max_workers: The maximum number of concurrent downloads. Default to 10.
    :param max_retries: The maximum number of retries if fails to download. Default to 5.
    :param target_path_func: A function that takes a URL as an argument and returns
    """

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # limit maximum number of workers to access to remote file server.
    semaphore = asyncio.Semaphore(max_workers)
    downloaded_files = []

    async def download_all():
        async def download_one(url: str):
            os.makedirs(dst_folder, exist_ok=True)
            # Use custom target_path_func if provided, otherwise default behavior
            if target_path_func is not None:
                target_path = target_path_func(url, dst_folder)

                new_dirname = os.path.dirname(target_path)
                if new_dirname != dst_folder:
                    os.makedirs(new_dirname, exist_ok=True)
                file_name = os.path.basename(target_path)
            else:
                file_name = os.path.basename(url)
                target_path = os.path.join(dst_folder, file_name)
            print("TARGET PATH FOLDER", target_path)
            for attempt in range(1, max_retries + 1):
                async with semaphore:
                    try:
                        await download_remote_file(
                            file_url=url,
                            output_file=target_path,
                            progress=progress)

                        downloaded_files.append(target_path)
                        return
                    except Exception as e:
                        logger.warning(f"[{file_name}] Attempt {attempt}/{max_retries} failed: {e}")
                        if attempt == max_retries:
                            logger.error(f"Failed to download {file_name} after {max_retries} attempts.")
                        else:
                            await asyncio.sleep(3)

        tasks = [
            asyncio.create_task(download_one(file_url))
            for file_url in file_urls
        ]
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.warning("Download interrupted by user. Cancelling tasks...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    loop.run_until_complete(download_all())
    loop.close()

    return downloaded_files


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


async def download_remote_files1(
        file_urls: Iterable[str],
        dst_folder: str,
        progress=None,
        target_path_func=None):
    """
    Download remote files from a list of URLs.

    :param file_urls: The URLs of the files to download.
    :param dst_folder: The folder to save the files in.
    :param progress: An optional rich progress bar instance.
    :param target_path_func: A function that takes a URL as an argument and returns
    """

    try:
        async with aiohttp.ClientSession() as client_session:
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
                    asyncio.create_task(download_s3_object(url=file_url, filename=target_path, progress=progress,
                                    client_session=client_session))
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    # raise the error as if download on even one fails, the results should not be collected
                    raise result
            return results
    except (KeyboardInterrupt, asyncio.CancelledError) as e:
        raise
    except Exception as e:
        logger.error(f"Failed to download remote files: {e}")
        raise

