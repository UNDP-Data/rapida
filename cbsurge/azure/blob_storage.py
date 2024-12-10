import logging
import os

import aiofiles
from azure.storage.blob.aio import ContainerClient
from tqdm.asyncio import tqdm

from cbsurge.constants import AZURE_BLOB_CONTAINER_NAME


class AzureBlobStorageManager:
    """
    A class to manage Azure Blob Storage operations.

    usage:
    async with AzureBlobStorageManager(connection_string) as az:
        await az.upload_blob(file_path, blob_name)
        await az.upload_files(local_directory, azure_directory)
        await az.download_blob(blob_name, local_directory)
        await az.download_files(azure_directory, local_directory)
    """
    def __aenter__(self):
        return self

    def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def __init__(self, conn_str):
        logging.info("Initializing Azure Blob Storage Manager")
        self.conn_str = conn_str
        self.container_client = ContainerClient.from_connection_string(conn_str=conn_str, container_name=AZURE_BLOB_CONTAINER_NAME)

    async def upload_blob(self, file_path=None, blob_name=None):
        """
        Upload a file to Azure Blob Storage.
        Args:
            file_path: (str) The path to the file to upload.
            blob_name: (str) The name of the blob to create, including the path relative to the container.

        Returns:

        """
        logging.info("Starting upload for blob: %s", blob_name)
        blob_client = self.container_client.get_blob_client(blob=blob_name)

        # Get the file size for progress tracking
        file_size = os.path.getsize(file_path)

        async def __progress__(current, total):
            tqdm(total=total, unit="B", unit_scale=True, desc=f"Uploading {blob_name}").update(current)

        async with aiofiles.open(file_path, "rb") as data:
            await blob_client.upload_blob(data, overwrite=True, max_concurrency=4, blob_type="BlockBlob", length=file_size, progress_hook=__progress__)

        logging.info("Upload completed for blob: %s", blob_name)
        return blob_client.url


    async def upload_files(self, local_directory=None, azure_directory=None):
        """
        Upload multiple files to Azure Blob Storage.
        Args:
            local_directory:
            azure_directory:

        Returns:
        """

        for root, _, files in os.walk(local_directory):
            for file in files:
                async def __progress__(current, total):
                    tqdm(total=total, unit="B", unit_scale=True, desc=f"Uploading {blob_name}").update(current)
                file_path = os.path.join(root, file)
                blob_name = f"{azure_directory}/{file}"
                await self.upload_blob(file_path=file_path, blob_name=blob_name)
        return

    async def download_blob(self, blob_name=None, local_directory=None):
        """
        Download a blob from Azure Blob Storage.
        Args:
            blob_name: (str) The name of the blob to download.
            local_directory: (str) The local path to save the downloaded blob. If not provided, the blob will be saved in the current working directory.
        Returns:

        """
        logging.info("Downloading blob: %s", blob_name)
        blob_client = self.container_client.get_blob_client(blob=blob_name)
        file_name = blob_name.split("/")[-1]
        if local_directory:
            os.makedirs(local_directory, exist_ok=True)
            file_name = f"{local_directory}/{file_name}"
        async with aiofiles.open(file_name, "wb") as data:
            blob = await blob_client.download_blob()
            await data.write(await blob.readall())
        logging.info("Download completed for blob: %s", blob_name)
        return file_name


    async def download_files(self, azure_directory=None, local_directory=None):
        """
        Download multiple blobs from Azure Blob Storage.
        Args:
            azure_directory: (str) The directory in Azure Blob Storage to download blobs from.
            local_directory: (str) The local path to save the downloaded blobs. If not provided, the blobs will be saved in the current working directory.
        Returns:

        """
        async for blob in self.container_client.list_blobs(name_starts_with=azure_directory):
            await self.download_blob(blob_name=blob.name, local_directory=local_directory)
        return


    async def close(self):
        """
        Close the Azure Blob Storage Manager.
        Returns:
        """
        logging.info("Closing Azure Blob Storage Manager")
        return await self.container_client.close()

