import logging
import os

import aiofiles
from azure.storage.blob.aio import BlobServiceClient
from tqdm.asyncio import tqdm

CONTAINER_NAME = "stacdata"

class AzStorageManager:
    def __init__(self, conn_str):
        logging.info("Initializing Azure Storage Manager")
        self.conn_str = conn_str
        self.blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        self.container_client = self.blob_service_client.get_container_client(CONTAINER_NAME)

    async def upload_file(self, file_path=None, blob_name=None):
        """
        Upload a file to Azure Blob Storage.
        Args:
            file_path: (str) The path to the file to upload
            blob_name: (str) The name of the blob to create

        Returns:

        """
        logging.info("Starting upload for blob: %s", blob_name)
        blob_client = self.container_client.get_blob_client(blob_name)

        # Get the file size for progress tracking
        file_size = os.path.getsize(file_path)

        async def __progress__(current, total):
            tqdm(total=total, unit="B", unit_scale=True, desc=f"Uploading {blob_name}").update(current)

        async with aiofiles.open(file_path, "rb") as data:
            await blob_client.upload_blob(data, overwrite=True, max_concurrency=4, blob_type="BlockBlob", length=file_size, progress_hook=__progress__)

        logging.info("Upload completed for blob: %s", blob_name)
        return blob_client.url

    async def close(self):
        logging.info("Closing Azure Storage Manager")
        await self.blob_service_client.close()
        await self.container_client.close()