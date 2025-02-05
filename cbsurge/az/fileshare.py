import asyncio
import logging
import os

from azure.storage.fileshare.aio import ShareFileClient

from cbsurge import util
from cbsurge.session import Session

logger = logging.getLogger(__name__)


async def download(file_client: ShareFileClient = None, dst_path: str = None, progress=None, task=None):
    """
    Download a file from Azure File Share asynchronously.

    :param file_client: An instance of ShareFileClient.
    :param dst_path: Local path to save the downloaded file.
    :param progress: Optional progress handler.
    :param task: Optional progress task.
    """
    try:
        properties = await file_client.get_file_properties()
        if properties is None:
            raise ValueError(f"File {file_client.file_path} does not exist")

        logger.debug(f"Going to download {file_client.file_path}")
        file_stream = await file_client.download_file()

        with open(dst_path, "wb") as f:
            async for chunk in file_stream.chunks():
                f.write(chunk)

        if progress and task is not None:
            progress.update(task, advance=1)
    except Exception as e:
        logger.error(f"Failed to download the file from {file_client.file_path}: {e}")
        raise
    else:
        logger.debug(f"Downloaded file {file_client.file_path} to {dst_path}")
        return dst_path


async def download_file(src_path: str = None, dst_path: str = None) -> str:
    """
    Download a file from Azure File Share to a local storage.

    Parameters
    ----------
    src_path : str
        Source path to download the file from.
    dst_path : str, optional
        Destination path to save the file to.
    Returns
    -------
    path_dst : str
        Destination path the file was downloaded to.
    """
    assert isinstance(dst_path, str), f'dst_path must be a valid string, not {type(dst_path)}'

    util.validate_azure_storage_path(a_path=src_path)

    proto, account_name, src_file_path = src_path.split(':')
    share_name, *src_path_parts = src_file_path.split(os.path.sep)
    rel_src_file_path = os.path.sep.join(src_path_parts)

    async with Session() as session:
        async with session.get_share_file_client(account_name=account_name, share_name=share_name, file_path=rel_src_file_path) as file_client:
            return await download(file_client=file_client, dst_path=dst_path)


async def upload_file(src_path, dst_path, file_client: ShareFileClient = None, overwrite=True):
    """
    Upload a file to Azure File Share asynchronously.

    :param src_path: str, path to the local file to be uploaded
    :param dst_path: str, the fully qualified path to a file in Azure in format az:{account}:{share}/path.ext
    :param file_client: instance of Azure Storage ShareFileClient
    :param overwrite: bool, if the file will be overwritten
    :return: str, the fully qualified path to a file in Azure in format az:{account}:{share}/path.ext
    """
    if not os.path.exists(src_path):
        raise ValueError(f"File {src_path} does not exist")
    try:
        logger.debug(f"Uploading {src_path} to {dst_path}")
        with open(src_path, mode="rb") as data:
            await file_client.upload_file(data, overwrite=overwrite)
    except Exception as e:
        logger.error(f"Failed to upload {dst_path} from {src_path}: {e}")
        raise
    else:
        logger.debug(f"File {src_path} was successfully uploaded to {dst_path}")
        return dst_path


async def upload_file_to_share(src_path: str = None, dst_path: str = None):
    """
    Upload a file to Azure File Share.
    :param src_path: str, path to the local file to be uploaded
    :param dst_path: str, the fully qualified path to a file in Azure in format az:{account}:{share}/path.ext
    :return: str, the fully qualified path to a file in Azure in format az:{account}:{share}/path.ext
    """
    assert isinstance(src_path, str), f'src_path must be a valid string, not {type(src_path)}'
    util.validate_azure_storage_path(a_path=dst_path)
    proto, account_name, dst_file_path = dst_path.split(':')
    share_name, *dst_path_parts = dst_file_path.split(os.path.sep)
    rel_dst_file_path = os.path.sep.join(dst_path_parts)

    async with Session() as session:
        async with session.get_share_file_client(account_name=account_name, share_name=share_name, file_path=rel_dst_file_path) as file_client:
            return await upload_file(src_path, dst_path, file_client)


async def download_files(src_files, dst_folder, max_at_once=10, progress=None):
    """
    Download multiple files from Azure File Share to a local storage.

    :param src_files: list of str, list of source paths to download files from
    :param dst_folder: str, local destination folder
    :param max_at_once: int, maximum concurrent downloads
    :param progress: optional, progress tracker
    :return: list of downloaded file paths
    """
    proto, account_name, src_file_path = src_files[0].split(':')
    share_name, *src_file_path_parts = src_file_path.split(os.path.sep)
    downloaded_files = []
    download_task = None

    async with Session() as session:
        async with session.get_share_service_client(account_name=account_name, share_name=share_name) as sc:
            if progress:
                download_task = progress.add_task(
                    description=f'[blue]Downloading {len(src_files)} files', total=len(src_files))

            download_tasks = {}
            for file in src_files:
                *_, file_path = file.split(':')
                _, *file_path_parts = file_path.split(os.path.sep)
                rel_file_path = os.path.sep.join(file_path_parts)
                file_name = file_path_parts[-1]
                file_client = sc.get_share_client(file_path=rel_file_path)
                dst_path = os.path.join(dst_folder, file_name)

                task = asyncio.create_task(
                    download(file_client=file_client, dst_path=dst_path, progress=progress, task=download_task),
                    name=file_name)
                download_tasks[file_name] = task
            done, pending = await asyncio.wait(download_tasks.values(), return_when=asyncio.ALL_COMPLETED)
            for done_task in done:
                try:
                    downloaded_file_path = await done_task
                    downloaded_files.append(downloaded_file_path)
                except Exception as e:
                    logger.error(f'Failed to download {done_task.get_name()}. {e}')

            for pending_task in pending:
                pending_task.cancel()
                try:
                    await pending_task
                except asyncio.CancelledError:
                    logger.debug(f'Download {pending_task.get_name()} was cancelled')

    if progress and download_task:
        progress.remove_task(download_task)
    return downloaded_files

if __name__ == "__main__":
    asyncio.run(download_file('az:undpgeohub:cbrapida/rus_f_0_2020_constrained_UNadj_cog.tif', 'downloaded_files'))