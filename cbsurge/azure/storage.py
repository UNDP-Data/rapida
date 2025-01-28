import logging
from cbsurge.session import Session
logger = logging.getLogger(__name__)

async def download_blob(src_path: str, dst_path: str | None = None) -> str:
    """
    Download a blob from Azure Blob Storage to a local storage.

    Parameters
    ----------
    src_path : str
        Source path to download the blob from.
    dst_path : str, optional
        Destination path to save the blob to. If not provided, saves to the blob using its original name
        to the current directory.

    Returns
    -------
    path_dst : str
        Destination path the blob was downloaded to.
    """

    logger.debug(f"Checking in the blob at {src_path} exists")
    blob_client = self.container_client.get_blob_client(blob=src_path)
    if not await blob_client.exists():
        raise ValueError(f"Blob at {src_path} does not exist")

    if isinstance(dst_path, str):
        pass
    elif dst_path is None:
        dst_path = blob_client.blob_name
    else:
        raise ValueError(
            f"path_dst must be either None or a valid string, not {type(dst_path)}"
        )

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