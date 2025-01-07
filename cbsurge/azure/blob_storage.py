import asyncio
import logging
import os

import aiofiles
from azure.storage.blob.aio import ContainerClient
from tqdm.asyncio import tqdm_asyncio, tqdm

from cbsurge.constants import AZURE_BLOB_CONTAINER_NAME
from cbsurge.exposure.population.constants import WORLDPOP_AGE_MAPPING


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
    def __init__(self, conn_str):
        logging.info("Initializing Azure Blob Storage Manager")
        self.conn_str = conn_str
        self.container_client = ContainerClient.from_connection_string(conn_str=conn_str, container_name=AZURE_BLOB_CONTAINER_NAME)

    async def __aenter__(self):
        self.container_client = ContainerClient.from_connection_string(conn_str=self.conn_str, container_name=AZURE_BLOB_CONTAINER_NAME)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.container_client:
            return await self.close()

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
            tqdm_asyncio(total=total, unit="B", unit_scale=True, desc=f"Uploading {blob_name}").update(current)

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
                    tqdm_asyncio(total=total, unit="B", unit_scale=True, desc=f"Uploading {blob_name}").update(current)
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
            str: The path to the downloaded file.
        """
        logging.info("Downloading blob: %s", blob_name)
        blob_client = self.container_client.get_blob_client(blob=blob_name)
        file_name = blob_name.split("/")[-1]
        if local_directory:
            os.makedirs(local_directory, exist_ok=True)
            file_name = f"{local_directory}/{file_name}"

        blob_properties = await blob_client.get_blob_properties()
        total_size = blob_properties.size

        async with aiofiles.open(file_name, "wb") as data:
            blob = await blob_client.download_blob()
            progress = tqdm_asyncio(total=total_size, unit="B", unit_scale=True, desc=file_name)
            async for chunk in blob.chunks():
                await data.write(chunk)
                progress.update(len(chunk))
            progress.close()
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
        blobs = await self.list_blobs(prefix=azure_directory)
        for blob in blobs:
            await self.download_blob(blob_name=blob, local_directory=local_directory)
        return

    async def list_blobs(self, prefix=None):
        """
        List blobs in the Azure Blob Storage container.
        Args:
            prefix: (str) The prefix to filter blobs by.

        Returns:
        """
        return [blob.name async for blob in self.container_client.list_blobs(name_starts_with=prefix)]


    async def copy_file(self, source_blob=None, destination_blob=None):
        """
        Copy a file from one blob to another.
        Args:
            source_blob: (str) The name of the source blob to copy.
            destination_blob: (str) The name of the destination blob to copy to.

        Returns:

        """
        logging.info("Copying blob: %s to %s", source_blob, destination_blob)
        source_blob_client = self.container_client.get_blob_client(blob=source_blob)
        destination_blob_client = self.container_client.get_blob_client(blob=destination_blob)
        await destination_blob_client.start_copy_from_url(source_blob_client.url)
        return destination_blob_client.url

    async def delete_blob(self, blob_name=None):
        """
        Delete a blob from Azure Blob Storage.
        Args:
            blob_name:

        Returns:

        """
        logging.info("Deleting blob: %s", blob_name)
        blob_client = self.container_client.get_blob_client(blob=blob_name)
        await blob_client.delete_blob()
        return blob_name

    async def rename_file(self, source_blob=None, destination_blob=None):
        """
        Rename a blob file
        Args:
            source_blob:
            destination_blob:

        Returns:

        """
        await self.copy_file(source_blob=source_blob, destination_blob=destination_blob)
        await self.delete_blob(blob_name=source_blob)
        return destination_blob

    async def close(self):
        """
        Close the Azure Blob Storage Manager.
        Returns:
        """
        logging.info("Closing Azure Blob Storage Manager")
        return await self.container_client.close()
