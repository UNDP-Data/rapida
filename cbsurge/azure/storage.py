import asyncio
import logging
import os
from cbsurge.session import Session
from cbsurge import util
from azure.storage.blob import BlobType
from azure.storage.blob.aio import BlobClient
logger = logging.getLogger(__name__)



async def download(src_path, dst_path, blob_client:BlobClient = None):

    if not await blob_client.exists():
        raise ValueError(f"Blob {src_path} does not exist")
    try:
        logger.debug(f"Downloading the blob from {dst_path}")
        blob = await blob_client.download_blob()
        async for chunk in blob.chunks():
            with open(dst_path, "wb") as f:
                f.write(chunk)
    except Exception as e:
        logger.error(f"Failed to download the blob from {src_path}: {e}")
    else:
        logger.debug(f"Downloaded blob {src_path} to {dst_path}")
        return dst_path


async def download_blob_with_session(session: Session | None = None, src_path: str | None = None, dst_path: str | None = None) -> str:
    """
    Download a blob from Azure Blob Storage to a local storage.

    Parameters
    ----------
    session: cbsurge.session.Session, instance of Session
    src_path : str
        Source path to download the blob from.
    dst_path : str, optional
        Destination path to save the blob to.
    Returns
    -------
    path_dst : str
        Destination path the blob was downloaded to.
    """
    assert isinstance(dst_path, str), f'dst_path must be a valid string, not {type(dst_path)}'

    util.validate_azure_storage_path(a_path=src_path)

    proto, account_name, dst_blob_path = src_path
    container_name, *src_path_parts = dst_blob_path.split(os.path.sep)
    rel_src_blob_path = os.path.sep.join(src_path_parts)

    logger.debug(f"Checking in the blob at {src_path} exists")
    async with session.get_blob_container_client(account_name=account_name, container_name=container_name) as cc:
        blob_client = cc.get_blob_client(blob=rel_src_blob_path)
        return await download(src_path, dst_path, blob_client)

async def download_blob(src_path: str | None = None, dst_path: str | None = None) -> str:
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

    proto, account_name, dst_blob_path = src_path.split(':')
    container_name, *src_path_parts = dst_blob_path.split(os.path.sep)
    rel_src_blob_path = os.path.sep.join(src_path_parts)
    async with Session() as session:
        logger.debug(f"Checking in the blob at {src_path} exists")
        async with session.get_blob_container_client(account_name=account_name,container_name=container_name) as cc:
            blob_client = cc.get_blob_client(blob=rel_src_blob_path)
            return await download(src_path, dst_path, blob_client)


async def upload(src_path, dst_path, blob_client:BlobClient|None = None, blob_type:BlobType =  BlobType.BLOCKBLOB, overwrite=True):
    """

    :param src_path: str, path to the local file to be uploaded
    :param dst_path: str, the fully qualified path to a blob in azure in format az:{account}:{container}/path.ext
    :param blob_client: instance of Azure Storage  BlobClient
    :param blob_type: one of BlobType.APPENDBLOB, BlobType.BLOCKBLOB, BlobType.PAGEBLOB
    :param overwrite:bool, if the blob will be overwritten
    :return: str, the fully qualified path to a blob in azure in format az:{account}:{container}/path.ext
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
            )
    except Exception as e:
        logger.error(f"Failed to upload {dst_path} from {src_path}: {e}")
    else:
        logger.debug(f"Blob {src_path} was successfully uploaded to {dst_path}")
        return dst_path






async def upload_blob(src_path:str|None=None, dst_path: str|None = None):
    """
    Upload a blob from a local file
    :param src_path: str, path to the local file to be uploaded
    :param dst_path: str, the fully qualified path to a blob in azure in format az:{account}:{container}/path.ext
    :return: str, the fully qualified path to a blob in azure in format az:{account}:{container}/path.ext

    """
    assert isinstance(src_path, str), f'src_path must be a valid string, not {type(src_path)}'
    util.validate_azure_storage_path(a_path=dst_path)
    proto, account_name, dst_blob_path = dst_path
    container_name, *dst_path_parts = dst_blob_path.split(os.path.sep)
    rel_dst_blob_path = os.path.sep.join(dst_path_parts)
    async with Session() as session:
        logger.debug(f"Checking in the blob at {src_path} exists")
        with session.get_blob_container_client(account_name=account_name,container_name=container_name) as cc:
            blob_client = cc.get_blob_client(blob=rel_dst_blob_path)
            return upload(src_path, dst_path, blob_client)

async def upload_blob_with_session(session:Session|None=None, src_path:str|None=None, dst_path: str|None = None):
    """
    Upload a blob from a local file
    :param src_path: str, path to the local file to be uploaded
    :param session, instance of cbsurge.session.Session,
    :param dst_path: str, the fully qualified path to a blob in azure in format az:{account}:{container}/path.ext
    :return: str, the fully qualified path to a blob in azure in format az:{account}:{container}/path.ext

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



if __name__ == '__main__':
    import tempfile
    logger = util.setup_logger(name='rapida', level=logging.INFO)
    country = 'MDA'
    year=2020
    src_path = f'az:undpgeohub:stacdata/worldpop/{year}/{country}/aggregate/{country}_total.tif'

    with tempfile.TemporaryDirectory(delete=False) as tdir:
        with tempfile.NamedTemporaryFile(suffix='.tif', dir=tdir, delete=False) as ttif:
            asyncio.run(download_blob(src_path=src_path, dst_path=ttif.name))
            print(ttif.name)