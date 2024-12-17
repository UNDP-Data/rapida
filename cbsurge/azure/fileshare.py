import logging
import os

import aiofiles
import azure.core.exceptions
from azure.storage.fileshare.aio import ShareClient
from tqdm.asyncio import tqdm_asyncio


class AzureFileShareManager:
    """
    A class to manage Azure File Share operations.


    usage:
    async with AzureFileShareManager(connection_string, share_name) as az:
        await az.upload_file(local_file_path, azure_file_path)
        await az.upload_files(local_directory)
        await az.download_file(file_name, download_path)
    """

    def __init__(self, connection_string, share_name):
        self.connection_string = connection_string
        self.share_name = share_name
        self.share_client = ShareClient.from_connection_string(connection_string, share_name)

    def __aenter__(self):
        return self

    def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    async def close(self):
        """
        Close the Azure File Share client
        Returns: None

        """
        await self.share_client.close()

    async def create_azure_directory(self, path):
        """
        Create a directory in the Azure File Share. If the directory already exists, it will not be created.

        Args:
            path: (str) The path to the directory to create.

        Returns:
            The fully created directory path.
        """
        directory_list = path.split("/")
        current_path = ""
        for directory in directory_list:
            current_path = f"{current_path}/{directory}" if current_path else directory
            directory_client = self.share_client.get_directory_client(current_path)
            try:
                await directory_client.create_directory()
                logging.info(f"Created directory: {current_path}")
            except azure.core.exceptions.ResourceExistsError:
                logging.info(f"Directory {current_path} already exists")
            except Exception as e:
                logging.error(f"Error creating directory {current_path}: {e}")
                raise e
        return current_path

    async def upload_file(self, local_file_path, azure_file_path):
        """
        Upload a file to the Azure File Share.

        Args:
            local_file_path: The path to the file on the local machine.
            azure_file_path: The path to the file in the Azure File Share.

        Returns:
            The Azure File Path after upload.
        """
        file_client = self.share_client.get_file_client(azure_file_path)

        async def __progress__(current, total):
            tqdm_asyncio(total=total, unit="B", unit_scale=True, desc=f"Uploading {azure_file_path}").update(current)

        # Ensure parent directories exist
        if "/" in azure_file_path:
            directory_path = "/".join(azure_file_path.split("/")[:-1])
            await self.create_azure_directory(directory_path)

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



    async def upload_files(self, local_directory):
        """
        Upload all files in a local directory supplied to the Azure File Share.
        Args:
            local_directory: (str) The local directory to upload.

        Returns:

        """
        for root, _, files in os.walk(local_directory):
            for file in files:
                await self.upload_file(os.path.join(root, file), file)
        return local_directory


    async def download_file(self, file_name, download_path):
        """
        Download a file from the Azure File Share.
        Args:
            file_name: The name of the file to download. This should include the path relative to the share.
            download_path: The local path to save the downloaded file.

        Returns:

        """
        file_client = self.share_client.get_file_client(file_name)
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

    async def copy_file(self, source_file, destination_file):
        """
        Copy a file from one location to another in the Azure File Share.
        Args:
            source_file: The file to copy.
            destination_file: The destination file.

        Returns:

        """
        source_file_client = self.share_client.get_file_client(source_file)
        destination_file_client = self.share_client.get_file_client(destination_file)

        await destination_file_client.start_copy_from_url(source_file_client.url)
        return destination_file

