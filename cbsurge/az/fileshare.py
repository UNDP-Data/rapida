import asyncio
import logging
import os

import aiofiles
from azure.storage.fileshare.aio import ShareFileClient
from azure.core.exceptions import ResourceNotFoundError
from tqdm.asyncio import tqdm_asyncio

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


async def create_azure_directory(path):
        """
        Create a directory in the Azure File Share. If the directory already exists, it will not be created.

        Args:
            path: (str) The path to the directory to create.

        Returns:
            The fully created directory path.
        """
        directory_list = path.split("/")
        current_path = ""
        async with Session() as session:
            async with session.get_share_client(account_name=session.get_account_name(), share_name=session.get_file_share_name()) as share_client:
                for directory in directory_list:
                    current_path = f"{current_path}/{directory}" if current_path else directory
                    directory_client = share_client.get_directory_client(current_path)
                    try:
                        await directory_client.create_directory()
                        logging.info(f"Created directory: {current_path}")
                    except ResourceNotFoundError:
                        logging.info(f"Directory {current_path} already exists")
                    except Exception as e:
                        logging.error(f"Error creating directory {current_path}: {e}")
                        raise e
                return current_path


async def download_file(file_name, download_path):
    """
    Download a file from the Azure File Share.
    Args:
        file_name: The name of the file to download. This should include the path relative to the share.
        download_path: The local path to save the downloaded file.

    Returns:

    """
    async with Session() as session:
        async with session.get_share_client(account_name=session.get_account_name(), share_name=session.get_file_share_name()) as share_client:
            file_client = share_client.get_file_client(file_name)
            file_properties = await file_client.get_file_properties()
            file_size = file_properties['content_length']

            progress_bar = tqdm_asyncio(total=file_size, unit="B", unit_scale=True, desc=f"Downloading {file_name}")

            async with aiofiles.open(os.path.join(download_path, file_name), "wb") as file:
                stream = await file_client.download_file()
                async for chunk in stream.chunks():
                    await file.write(chunk)
                    progress_bar.update(len(chunk))

            progress_bar.close()
            return file_name


# async def upload_file(src_path, dst_path, file_client: ShareFileClient = None, overwrite=True):
#     """
#     Upload a file to Azure File Share asynchronously.
#
#     :param src_path: str, path to the local file to be uploaded
#     :param dst_path: str, the fully qualified path to a file in Azure in format az:{account}:{share}/path.ext
#     :param file_client: instance of Azure Storage ShareFileClient
#     :param overwrite: bool, if the file will be overwritten
#     :return: str, the fully qualified path to a file in Azure in format az:{account}:{share}/path.ext
#     """
#     if not os.path.exists(src_path):
#         raise ValueError(f"File {src_path} does not exist")
#     try:
#         logger.debug(f"Uploading {src_path} to {dst_path}")
#         with open(src_path, mode="rb") as data:
#             await file_client.upload_file(data, overwrite=overwrite)
#     except Exception as e:
#         logger.error(f"Failed to upload {dst_path} from {src_path}: {e}")
#         raise
#     else:
#         logger.debug(f"File {src_path} was successfully uploaded to {dst_path}")
#         return dst_path


async def upload_file(local_file_path, azure_file_path):
    """
    Upload a file to the Azure File Share.

    Args:
        local_file_path: The path to the file on the local machine.
        azure_file_path: The path to the file in the Azure File Share.

    Returns:
        The Azure File Path after upload.
    """
    async with Session() as session:
        async with session.get_share_client(account_name=session.get_account_name(), share_name=session.get_file_share_name()) as share_client:
            file_client = share_client.get_file_client(azure_file_path)

            async def __progress__(current, total):
                tqdm_asyncio(total=total, unit="B", unit_scale=True, desc=f"Uploading {azure_file_path}").update(current)

            # Ensure parent directories exist
            if "/" in azure_file_path:
                directory_path = "/".join(azure_file_path.split("/")[:-1])
                await create_azure_directory(directory_path)

            # Upload the file
            async with aiofiles.open(local_file_path, "rb") as file:
                try:
                    file_size = os.path.getsize(local_file_path)
                    await file_client.upload_file(file, max_concurrency=4, length=file_size, progress_hook=__progress__)
                    logging.info(f"Uploaded file: {azure_file_path}")
                except Exception as e:
                    logging.error(f"Error uploading file {azure_file_path}: {e}")
                    raise e

                return azure_file_path


    async def upload_files(local_directory):
        """
        Upload all files in a local directory supplied to the Azure File Share.
        Args:
            local_directory: (str) The local directory to upload.

        Returns:

        """
        for root, _, files in os.walk(local_directory):
            for file in files:
                await upload_file(os.path.join(root, file), file)
        return local_directory



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
    asyncio.run(download_file('az:undpgeohub:cbrapida/Myanmar flood example/FL20240912MMR_SHP.zip', 'downloaded_files'))