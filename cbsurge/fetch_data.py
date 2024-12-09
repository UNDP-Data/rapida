# fetch data from azure. if data is not found, download it from the source

import logging

import httpx

class WorldPopFetcher:
    def __init__(self):
        self.file_name = None
        self.year = 2020
        self.country = None
        self.azure_root_url = "https://undpgeohub.blob.core.windows.net/stacdata/worldpop"
        self.worldpop_root_url = "https://hub.worldpop.org/geodata"

    def __construct_url__(self):
        """
        Construct the URL to fetch the data from.
        Returns:
        """
        assert self.file_name or (self.year and self.country), "Either file_name or year and country must be provided"

        if not self.file_name:
            return f"{self.worldpop_root_url}/{self.year}/{self.country}/"
        return f"{self.azure_root_url}/{self.year}/{self.country}/{self.file_name}"


    def fetch_data(self, file_name=None, url=None):
        """
        Fetch data from Azure Blob Storage. If data is not found, download it from the source.
        Returns:
        """
        assert file_name or url, "Either file_name or url must be provided"

        if not file_name:
            file_name = self.file_name
        logging.info("Fetching data from Azure Blob Storage")
        return file_name



def main():
    worldpop_fetcher = WorldPopFetcher()
    file = worldpop_fetcher.fetch_data(file_name="file_name")
    print(file)

if __name__ == "__main__":
    main()