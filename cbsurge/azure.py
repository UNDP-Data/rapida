import logging
import os

import aiofiles
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.fileshare.aio import ShareServiceClient
from rasterio.rio.blocks import blocks

from tqdm.asyncio import tqdm

CONTAINER_NAME = "stacdata"
FILE_SHARE_NAME = "cbrapida"

class AzStorageManager:

    # def __aenter__(self):
    #     return self
    #
    # def __aexit__(self, exc_type, exc_val, exc_tb):
    #     return self.close()

    def __init__(self, conn_str):
        logging.info("Initializing Azure Storage Manager")
        self.conn_str = conn_str
        self.blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        self.share_service_client = ShareServiceClient.from_connection_string(conn_str)


    async def upload_blob(self, file_path=None, blob_name=None):
        """
        Upload a file to Azure Blob Storage.
        Args:
            file_path: (str) The path to the file to upload
            blob_name: (str) The name of the blob to create

        Returns:

        """

        container_client = self.blob_service_client.get_container_client(CONTAINER_NAME)
        logging.info("Starting upload for blob: %s", blob_name)
        blob_client = container_client.get_blob_client(blob_name)

        # Get the file size for progress tracking
        file_size = os.path.getsize(file_path)

        async def __progress__(current, total):
            tqdm(total=total, unit="B", unit_scale=True, desc=f"Uploading {blob_name}").update(current)

        async with aiofiles.open(file_path, "rb") as data:
            await blob_client.upload_blob(data, overwrite=True, max_concurrency=4, blob_type="BlockBlob", length=file_size, progress_hook=__progress__)

        logging.info("Upload completed for blob: %s", blob_name)
        return blob_client.url, await container_client.close()



    async def upload_files(self, local_directory=None, azure_directory=None):
        """
        Upload multiple files to Azure Blob Storage.
        Args:
            local_directory: (str) The local directory containing the files to upload
            azure_directory: (str) The directory in Azure Blob Storage to upload the files to

        Returns:

        """


        container_client = self.blob_service_client.get_container_client(CONTAINER_NAME)
        for root, _, files in os.walk(local_directory):
            for file in files:
                async def __progress__(current, total):
                    tqdm(total=total, unit="B", unit_scale=True, desc=f"Uploading {blob_name}").update(current)
                local_file_path = os.path.join(root, file)
                blob_name = os.path.join(azure_directory, file)
                logging.info(f"Uploading {local_file_path} as {blob_name}...")
                async with aiofiles.open(local_file_path, "rb") as data:
                    await container_client.upload_blob(name=blob_name, data=data, overwrite=True, max_concurrency=4, progress_hook=__progress__)
        logging.info("Upload completed for all files")
        return await container_client.close()



    async def upload_to_fileshare(self, local_file=None, file_name=None):
        """
        Upload a file to Azure File Share.
        Args:
            file_name:

        Returns:

        """

        async def __progress__(current, total):
            tqdm(total=total, unit="B", unit_scale=True, desc=f"Uploading {file_name}").update(current)
        share_client = self.share_service_client.get_share_client("cbrapida")
        file_client = share_client.get_file_client(file_name)
        logging.info("Starting upload for file: %s", file_name)
        async with aiofiles.open(file_path, "rb") as data:
            await file_client.upload_file(data, max_concurrency=4, length=os.path.getsize(file_path), progress_hook=__progress__)
        logging.info("Upload completed for file: %s", file_name)
        return file_client.url, await share_client.close()

    async def download_blob(self, blob_name=None, save_as=None):
        """
        Download a blob from Azure Blob Storage.
        Args:
            save_as: (str) The path to save the downloaded blob
            blob_name: (str) The name of the blob to download

        Returns:

        """
        container_client = self.blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        logging.info("Starting download for blob: %s", blob_name)
        async with aiofiles.open(save_as, "wb") as data:
            blob = await blob_client.download_blob()
            await data.write(await blob.readall())
        logging.info("Download completed for blob: %s", blob_name)

    async def download_from_fileshare(self, file_name=None, save_as=None):
        """
        Download a file from Azure File Share.
        Args:
            file_name: (str) The name of the file to download
            save_as: (str) The path to save the downloaded file
        """
        share_client = self.share_service_client.get_share_client(FILE_SHARE_NAME)
        file_client = share_client.get_file_client(file_name)
        logging.info("Starting download for file: %s", file_name)
        async with aiofiles.open(save_as, "wb") as data:
            file = await file_client.download_file()
            await data.write(await file.readall())
        logging.info("Download completed for file: %s", file_name)


    async def close(self):
        logging.info("Closing Azure Storage Manager")
        await self.blob_service_client.close()
        await self.share_service_client.close()