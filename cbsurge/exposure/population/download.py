import logging
import os
import tempfile
from asyncio import subprocess

import aiofiles

import httpx
from azure.storage.blob.aio import BlobServiceClient
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

load_dotenv()


ROOT_URL = "https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020_Constrained_UNadj/2020/"
AGE_MAPPING = {
    "0-12": 0,
    "1-4": 1,
    "5-9": 5,
    "10-14": 10,
    "15-19": 15,
    "20-24": 20,
    "25-29": 25,
    "30-34": 30,
    "35-39": 35,
    "40-44": 40,
    "45-49": 45,
    "50-54": 50,
    "55-59": 55,
    "60-64": 60,
    "65-69": 65,
    "70-74": 70,
    "75-79": 75,
    "80+": 80
}


class AzStorageManager:
    def __init__(self, conn_str):
        self.conn_str = conn_str
        self.blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        self.container_client = self.blob_service_client.get_container_client("worldpop")

    async def upload_file(self, file_path=None, blob_name=None):
        blob_client = self.container_client.get_blob_client(blob_name)

        # Get the file size for progress tracking
        file_size = os.path.getsize(file_path)
        chunk_size = 4 * 1024 * 1024  # 4MB chunks
        progress = tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Uploading {blob_name}")

        async with aiofiles.open(file_path, "rb") as data:
            while True:
                chunk = await data.read(chunk_size)
                if not chunk:
                    break
                await blob_client.upload_blob(chunk, overwrite=True, length=len(chunk),
                                              raw_response_hook=lambda _: progress.update(len(chunk)))

        progress.close()
        return blob_client.url

    async def close(self):
        await self.blob_service_client.close()
        await self.container_client.close()


def chunker_function(iterable, chunk_size=4):
    """
    Split an iterable into chunks of the specified size.
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


async def list_available_countries():
    url = "https://hub.worldpop.org/rest/data/pop/wpgpunadj"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=600)
            if response.status_code != 200:
                logging.error("Failed to get available countries, status: %s", response.status_code)
                return []
            res = response.json()
            data = res.get("data", [])
            return list(set([d.get("iso3") for d in data if d.get("iso3")]))
    except Exception as e:
        logging.error("Error fetching available countries: %s", e)
        return []


async def process_single_file(country_code=None, file_name=None, storage_manager=None):
    """
    Download a single file, convert the file to a COG and upload it to Azure Blob Storage.
    Args:
        country_code: Country code
        file_name: File name
        storage_manager: AzStorageManager instance

    Returns: None

    """
    url = f"{ROOT_URL}/{country_code}/{file_name}"
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url, timeout=600) as response:
                if response.status_code != 200:
                    logging.error("Failed to download file: %s, status: %s", url, response.status_code)
                    return

                total_size = int(response.headers.get("content-length", 0))
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {file_name}") as progress:
                    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
                        temp_file = f"{temp_dir}/{file_name}"
                        async with aiofiles.open(temp_file, 'wb') as f:
                            async for chunk in response.aiter_bytes(chunk_size=1024):
                                await f.write(chunk)
                                progress.update(len(chunk))
                            await f.close()
                            cog_path = f"{temp_dir}/{file_name.replace('.tif', '_cog.tif')}"
                            await convert_to_cog(input_file=temp_file, output_file=cog_path)
                            await storage_manager.upload_file(file_path=cog_path, blob_name=f"{os.getenv("AZ_ROOT_FILE_PATH")}/{country_code}/{file_name}")
                            os.unlink(temp_file)
                            os.unlink(cog_path)
        logging.info("Successfully downloaded: %s", file_name)
    except Exception as e:
        logging.error("Error downloading file %s: %s", url, e)





async def generate_list_of_files(country_code):
    """
    Generate a list of files to download for a given country.
    """
    file_list = []
    for sex in ['m', 'f']:
        for age_range in AGE_MAPPING.keys():
            file_name = f"{country_code.lower()}_{sex}_{AGE_MAPPING[age_range]}_2020_constrained_UNadj.tif"
            file_list.append(file_name)
    return file_list


from tqdm.asyncio import tqdm


async def download_data():
    countries = await list_available_countries()
    countries = ["KEN"]
    storage_manager = AzStorageManager(conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    tasks = []
    for country in countries:
        files = await generate_list_of_files(country_code=country)
        for i, file_chunk in enumerate(chunker_function(files, chunk_size=4)):
            if i > 0:
                break
            for file in file_chunk:
                download_task = process_single_file(country_code=country, file_name=file, storage_manager=storage_manager)
                tasks.append(download_task)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for results in results:
            if isinstance(results, Exception):
                logging.error("Error processing file: %s", results)
    #  Close the storage manager connection after all files have been uploaded
    await storage_manager.close()



async def convert_to_cog(input_file=None, output_file=None):
    """
    Convert a GeoTIFF file to a COG.
    Args:
        input_file: Input file path
        output_file: Output file path
    """

    cmd = ["gdal_translate",
           "-of", "COG",
           "-co", "OVERVIEW_RESAMPLING=NEAREST",
           "-co", "TARGET_SRS=EPSG:3857",
           "-co", "COMPRESS=ZSTD",
           "-co", "RESAMPLING=NEAREST",
           input_file, output_file]

    proc = await subprocess.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    await proc.communicate()
    return output_file


if __name__ == "__main__":
    import asyncio
    asyncio.run(download_data())
