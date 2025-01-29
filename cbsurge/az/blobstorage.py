import asyncio
import logging
import os
from csv import excel

from cbsurge.session import Session
from cbsurge import util
from azure.storage.blob import BlobType
from azure.storage.blob.aio import BlobClient
from typing import Iterable
from rich.progress import Progress
logger = logging.getLogger(__name__)



async def download(blob_client:BlobClient = None, dst_path:str=None, ):

    if not await blob_client.exists():
        raise ValueError(f"Blob {blob_client.blob_name} does not exist")
    try:
        logger.debug(f"Going to download {blob_client.blob_name}")
        blob = await blob_client.download_blob()
        async for chunk in blob.chunks():
            with open(dst_path, "wb") as f:
                f.write(chunk)

    except Exception as e:
        logger.error(f"Failed to download the blob from {blob_client.blob_name}: {e}")
    else:
        logger.debug(f"Downloaded blob {blob_client.blob_name} to {dst_path}")
        return dst_path


async def download_blob_with_session(session: Session  = None, src_path: str  = None, dst_path: str  = None) -> str:
    """
    Download a blob from Azure Blob Storage to a local storage.

    Parameters
    ----------
    session: cbsurge.session.Session, instance of Session
    src_path : str
        Source path to download the blob from.
    dst_path : str,
        Destination path to save the blob to.
    Returns
    -------
    path_dst : str
        Destination path the blob was downloaded to.
    """
    assert isinstance(dst_path, str), f'dst_path must be a valid string, not {type(dst_path)}'

    util.validate_azure_storage_path(a_path=src_path)
    proto, account_name, src_blob_path = src_path.split(':')

    container_name, *src_path_parts = src_blob_path.split(os.path.sep)
    rel_src_blob_path = os.path.sep.join(src_path_parts)

    logger.debug(f"Checking in the blob at {src_path} exists")
    async with session.get_blob_container_client(account_name=account_name, container_name=container_name) as cc:
        blob_client = cc.get_blob_client(blob=rel_src_blob_path)
        return await download(blob_client=blob_client, dst_path=dst_path )

async def download_blob(src_path: str = None, dst_path: str = None) -> str:
    """
    Download a blob from Azure Blob Storage to a local storage.

    Parameters
    ----------
    src_path : str
        Source path to download the blob from.
    dst_path : str, optional
        Destination path to save the blob to
    Returns
    -------
    path_dst : str
        Destination path the blob was downloaded to.
    """
    assert isinstance(dst_path, str), f'dst_path must be a valid string, not {type(dst_path)}'

    util.validate_azure_storage_path(a_path=src_path)

    proto, account_name, src_blob_path = src_path.split(':')
    container_name, *src_path_parts = src_blob_path.split(os.path.sep)
    rel_src_blob_path = os.path.sep.join(src_path_parts)
    async with Session() as session:
        async with session.get_blob_container_client(account_name=account_name,container_name=container_name) as cc:
            blob_client = cc.get_blob_client(blob=rel_src_blob_path)
            return await download(blob_client=blob_client, dst_path=dst_path )


async def upload(src_path, dst_path, blob_client:BlobClient = None, blob_type:BlobType =  BlobType.BLOCKBLOB, overwrite=True):
    """

    :param src_path: str, path to the local file to be uploaded
    :param dst_path: str, the fully qualified path to a blob in az in format az:{account}:{container}/path.ext
    :param blob_client: instance of Azure Storage  BlobClient
    :param blob_type: one of BlobType.APPENDBLOB, BlobType.BLOCKBLOB, BlobType.PAGEBLOB
    :param overwrite:bool, if the blob will be overwritten
    :return: str, the fully qualified path to a blob in az in format az:{account}:{container}/path.ext
    """
    if not os.path.exists(src_path):
        raise ValueError(f"File {src_path} does not exist")
    try:
        logger.debug(f"Uploading {src_path} to {dst_path}")
        with open(src_path, mode="rb") as data:
            await blob_client.upload_blob(
                data=data,
                overwrite=overwrite,
                blob_type = blob_type,
                #max_concurrency=8
            )
    except Exception as e:
        logger.error(f"Failed to upload {dst_path} from {src_path}: {e}")
    else:
        logger.debug(f"Blob {src_path} was successfully uploaded to {dst_path}")
        return dst_path






async def upload_blob(src_path: str = None, dst_path: str = None):
    """
    Upload a blob from a local file
    :param src_path: str, path to the local file to be uploaded
    :param dst_path: str, the fully qualified path to a blob in az in format az:{account}:{container}/path.ext
    :return: str, the fully qualified path to a blob in az in format az:{account}:{container}/path.ext

    """
    assert isinstance(src_path, str), f'src_path must be a valid string, not {type(src_path)}'
    util.validate_azure_storage_path(a_path=dst_path)
    proto, account_name, dst_blob_path = dst_path
    container_name, *dst_path_parts = dst_blob_path.split(os.path.sep)
    rel_dst_blob_path = os.path.sep.join(dst_path_parts)
    async with Session() as session:
        with session.get_blob_container_client(account_name=account_name,container_name=container_name) as cc:
            blob_client = cc.get_blob_client(blob=rel_dst_blob_path)
            return upload(src_path, dst_path, blob_client)

async def upload_blob_with_session(session:Session=None, src_path:str=None, dst_path: str = None):
    """
    Upload a blob from a local file
    :param src_path: str, path to the local file to be uploaded
    :param session, instance of cbsurge.session.Session,
    :param dst_path: str, the fully qualified path to a blob in az in format az:{account}:{container}/path.ext
    :return: str, the fully qualified path to a blob in az in format az:{account}:{container}/path.ext

    """
    assert isinstance(src_path, str), f'src_path must be a valid string, not {type(src_path)}'
    util.validate_azure_storage_path(a_path=dst_path)
    proto, account_name, dst_blob_path = dst_path
    container_name, *dst_path_parts = dst_blob_path.split(os.path.sep)
    rel_dst_blob_path = os.path.sep.join(dst_path_parts)

    logger.debug(f"Checking in the blob at {src_path} exists")
    with session.get_blob_container_client(account_name=account_name,container_name=container_name) as cc:
        blob_client = cc.get_blob_client(blob=rel_dst_blob_path)
        return upload(src_path, dst_path, blob_client)

async def download_blobs(src_blobs:Iterable[str] = None, dst_folder = None, max_at_once=10):
    proto, account_name, src_blob_path = src_blobs[0].split(':')
    container_name, *src_blob_path_parts = src_blob_path.split(os.path.sep)
    downloaded_files = list()
    async with Session() as session:
        async with session.get_blob_container_client(account_name=account_name,container_name=container_name) as cc:
            with Progress() as progress:
                total_task = progress.add_task(
                    description=f'[red]Going to download {len(src_blobs)} blobs', total=len(src_blobs))

                for i, blobs in enumerate(util.chunker(src_blobs, size=max_at_once), start=1):
                    download_tasks = dict()
                    try:

                        for src_blob in blobs:
                            *_, src_blob_path = src_blob.split(':')
                            _, *src_blob_path_parts = src_blob_path.split(os.path.sep)
                            rel_src_blob_path = os.path.sep.join(src_blob_path_parts)
                            src_blob_name = src_blob_path_parts[-1]
                            blob_client = cc.get_blob_client(blob=rel_src_blob_path)
                            dst_path = os.path.join(dst_folder, src_blob_name)

                            task = asyncio.create_task(
                                download(blob_client=blob_client, dst_path=dst_path),
                                name=src_blob_name
                            )
                            download_tasks[src_blob_name] = task

                        if not download_tasks: continue
                        done, pending = await asyncio.wait(download_tasks.values(),
                                                           timeout=90 * len(download_tasks),
                                                           return_when=asyncio.ALL_COMPLETED)
                        failed_tasks = []
                        for done_task in done:
                            try:
                                downloaded_file_name = task.get_name()
                                downloaded_file_path = await done_task
                                assert os.path.exists(downloaded_file_path), f'{downloaded_file_path} does not exist'
                                downloaded_files.append(downloaded_file_path)
                                progress.update(total_task,
                                                description=f'[red]Downloaded  {len(downloaded_files)} out of {len(src_blobs)} ',
                                                advance=1)
                            except Exception as e:
                                logger.error(f'Failed to download tile {downloaded_file_name}. {e}')
                                failed_tasks.append((downloaded_file_name, e))
                        # handle pending
                        for pending_task in pending:
                            try:
                                filename_to_download = pending_task.get_name()
                                pending_task.cancel()
                                await pending_task
                                failed_tasks.append((downloaded_file_name, asyncio.Timeout))
                            except Exception as e:
                                logger.error(f'Failed to download tile {filename_to_download}. {e}')
                                failed_tasks.append((filename_to_download, e))
                    except (asyncio.CancelledError, KeyboardInterrupt) as de:
                        logger.debug(f'Cancelling download tasks')
                        for filename_to_download, t in download_tasks.items():
                            if t.done(): continue
                            if not t.cancelled():
                                try:
                                    t.cancel()
                                    await t
                                except asyncio.CancelledError:
                                    logger.debug(f'Download {filename_to_download} was cancelled')
                        if de.__class__ == KeyboardInterrupt:
                            raise de
                        else:
                            break

    return downloaded_files

# if __name__ == '__main__':
#     import tempfile
#     logger = util.setup_logger(name='rapida', level=logging.INFO)
#     country = 'MDA'
#     year=2020
#     src_path = f'az:undpgeohub:stacdata/worldpop/{year}/{country}/aggregate/{country}_total.tif'
#
#     with tempfile.TemporaryDirectory(delete=False) as tdir:
#         with tempfile.NamedTemporaryFile(suffix='.tif', dir=tdir, delete=False) as ttif:
#             asyncio.run(download_blob(src_path=src_path, dst_path=ttif.name))
#             print(ttif.name)