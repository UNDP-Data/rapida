import asyncio
import logging
import os
import shutil
import tempfile
from asyncio import subprocess
from html.parser import HTMLParser

import aiofiles
import httpx

from tqdm.asyncio import tqdm

from cbsurge.azure_upload import AzStorageManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("azure").setLevel(logging.WARNING)

CONTAINER_NAME = "stacdata"
AZ_ROOT_FILE_PATH = "worldpop"


class LinkExtractor(HTMLParser):
    """
    Extracts all href links from <a> tags within <td> elements, skipping <div class="title mb-3">.
    """
    def __init__(self):
        super().__init__()
        self.in_td = False
        self.links = []
        self.skip_table = False

    def handle_starttag(self, tag, attrs):
        if tag == 'div' and ('class', 'title mb-3') in attrs:
            self.skip_table = True

        if not self.skip_table:
            if tag == 'td':
                self.in_td = True
            elif tag == 'a' and self.in_td:
                for attr_name, attr_value in attrs:
                    if attr_name == 'href':
                        self.links.append(attr_value)

    def handle_endtag(self, tag):
        """
        Args:
            tag: The tag to handle

        Returns:

        """
        if tag == 'div' and self.skip_table:
            self.skip_table = False

        if tag == 'td':
            self.in_td = False

def chunker_function(iterable, chunk_size=4):
    """
    Split an iterable into chunks of the specified size.
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


async def get_available_data(country_code=None, year="2020"):
    """
    Args:
        country_code: The country code for which to fetch data
        year: (Not implemented) The year for which to fetch data

    Returns:

    """
    url = f"https://hub.worldpop.org/rest/data/age_structures/ascicua_{year}"
    logging.info("Fetching available countries from: %s", url)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=600)
            if response.status_code != 200:
                logging.error("Failed to get ava ilable countries, status: %s", response.status_code)
                return []
            res = response.json()
            data = res.get("data", [])
            if country_code:
                return {d.get("iso3"): d.get("id") for d in data if d.get("iso3") == country_code}
            return {d.get("iso3"): d.get("id") for d in data}
    except Exception as e:
        logging.error("Error fetching available countries: %s", e)
        return []


async def extract_links_from_table(html_text: str):
    """
    Extracts all href links from <a> tags within <td> elements, skipping <div class="title mb-3">.

    :param html_text: The HTML document as a string.
    :return: A list of links found in <a> tags within <td> but not within <div class="title mb-3">
    """
    parser = LinkExtractor()
    parser.feed(html_text)
    return parser.links


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
           "-co", "BLOCKSIZE=256",
           "-co", "BIGTIFF=YES",
           "-co", "WARP_RESAMPLING=NEAREST",
           "-co", "RESAMPLING=NEAREST",
           "-co", "STATISTICS=YES",
           input_file, output_file]

    proc = await subprocess.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    await proc.communicate()
    return output_file


async def validate_cog(file_path=None):
    """
    Validate a COG file using the `rio cogeo validate` command.
    Args:
        file_path:(str) Path to the COG file

    Returns: None

    """
    logging.info("Validating COG: %s", file_path)
    cmd = ["rio", "cogeo", "validate", file_path]
    proc = await subprocess.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if stderr:
        logging.error("Error validating COG: %s", stderr)
        raise Exception(f"Error validating COG: {stderr}")
    else:
        logging.info("COG validation successful: %s", stdout)



async def check_cog_exists(storage_manager=None, blob_path=None):
    """
    Validate that a COG file exists and is valid.
    Args:
        storage_manager: AzStorageManager instance
        blob_path: Path to the COG file
    """
    assert storage_manager is not None, "storage_manager is required"
    assert blob_path is not None, "blob_path is required"
    logging.info("Checking if COG exists: %s", blob_path)
    blob_client = storage_manager.container_client.get_blob_client(blob_path)
    return await blob_client.exists()



async def process_single_file(file_url=None, storage_manager=None, country_code=None, year=None, force_reprocessing=False, download_path=None):
    """
    Download a single file, convert the file to a COG and upload it to Azure Blob Storage.
    Args:
        force_reprocessing: Force reprocessing of the file even if it already exists in Azure
        year: The year for which the file is being processed
        country_code: The country code for which the file is being processed
        file_url: URL to download the file from
        download_path: (Path) The local path to download the COG files to. It will upload to Azure if not provided.
        storage_manager: AzStorageManager instance
    Returns: None

    """
    assert file_url is not None, "file_url is required"
    assert storage_manager is not None, "storage_manager is required"
    assert file_url.endswith(".tif"), "Only .tif files are supported"

    file_name = file_url.split("/")[-1]

    try:
        async with httpx.AsyncClient() as client:
            cog_exists = await check_cog_exists(storage_manager=storage_manager,
                                                blob_path=f"{AZ_ROOT_FILE_PATH}/{year}/{country_code}/{file_name}")
            if cog_exists and not force_reprocessing:
                logging.info("COG already exists in Azure, skipping upload: %s", file_name)
                return
            async with client.stream("GET", file_url, timeout=600) as response:
                if response.status_code != 200:
                    logging.error("Failed to download file: %s, status: %s", file_url, response.status_code)
                    return
                total_size = int(response.headers.get("content-length", 0))
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {file_name}") as progress:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_file_path = f"{temp_dir}/{file_name}"
                        logging.debug("Temporary file path: %s", temp_file_path)
                        async with aiofiles.open(temp_file_path, 'wb') as f:
                            async for chunk in response.aiter_bytes(chunk_size=1024):
                                await f.write(chunk)
                                progress.update(len(chunk))
                            await f.close()
                            cog_path = f"{temp_dir}/{file_name.replace('.tif', '_cog.tif')}"
                            logging.info("Converting file to COG: %s", cog_path)
                            await convert_to_cog(input_file=temp_file_path, output_file=cog_path)
                            await validate_cog(file_path=cog_path)
                            if download_path:
                                try:
                                    if not os.path.exists(download_path):
                                        logging.info("Download path does not exist. Creating download path: %s", download_path)
                                        os.makedirs(download_path, exist_ok=True)
                                    logging.info("Copying COG file locally: %s", cog_path)
                                    os.makedirs(f"{download_path}/{year}/{country_code}", exist_ok=True)
                                    shutil.move(cog_path, f"{download_path}/{year}/{country_code}/{file_name}")
                                    logging.info("Successfully copied COG file: %s", f"{download_path}/{year}/{country_code}/{file_name}")
                                except Exception as e:
                                    raise Exception(f"Error copying COG file: {e}")
                            else:
                                logging.info("Uploading COG file to Azure: %s", cog_path)
                                await storage_manager.upload_file(file_path=cog_path, blob_name=f"{AZ_ROOT_FILE_PATH}/{year}/{country_code}/{file_name}")

        logging.info("Successfully processed file: %s", file_name)
    except Exception as e:
        logging.error("Error processing file %s: %s", file_name, e)


async def get_links_from_table(data_id=None):
    """
    Args:
        data_id: (str) The data ID for which to fetch links from the table

    Returns:

    """
    url = f"https://hub.worldpop.org/geodata/summary?id={data_id}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            logging.error("Failed to fetch data from: %s, status: %s", url, response.status_code)
            return []
        return await extract_links_from_table(response.text)


async def download_data(country_code=None, year="2020", force_reprocessing=False, download_path=None):
    """
    Download all available data for a given country and year.
    Args:
        force_reprocessing: (bool) Force reprocessing of any files found in azure
        country_code: (str) The country code for which to download data
        year: (Not implemented) The year for which to download data
        download_path: (Path) The local path to download the COG files to. It will upload to Azure if not provided.

    Returns: None

    """
    logging.info("Starting data download")
    available_data = await get_available_data(country_code=country_code, year=year)
    storage_manager = AzStorageManager(conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    for country_code, country_id in available_data.items():
        # if country_code == "RUS":
        #     continue
        logging.info("Processing country: %s", country_code)
        file_links = await get_links_from_table(data_id=country_id)
        for i, file_urls_chunk in enumerate(chunker_function(file_links, chunk_size=4)):
            # if i > 0:
            #     break
            logging.info("Processing chunk %d for country: %s", i + 1, country_code)
            # Create a fresh list of tasks for each file chunk
            tasks = [process_single_file(file_url=url,
                                         storage_manager=storage_manager,
                                         year=year,
                                         country_code=country_code,
                                         force_reprocessing=force_reprocessing,
                                         download_path=download_path) for url in file_urls_chunk]

            # Gather tasks and process results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error("Error processing file: %s", result)

    # Close the storage manager connection after all files have been processed
    logging.info("Closing storage manager after processing all files")
    await storage_manager.close()

if __name__ == "__main__":
    asyncio.run(download_data(force_reprocessing=False))