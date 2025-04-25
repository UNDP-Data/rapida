import logging
import os
import json
from azure.core.exceptions import ClientAuthenticationError
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from rapida.az.surgeauth import SurgeTokenCredential

logger = logging.getLogger(__name__)



class Session(object):

    _instance = None  # Stores the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        constructor
        """
        self.config = self.get_config()



    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    def get_config_file_path(self) -> str:
        user_dir = os.path.expanduser("~")
        config_file_path = os.path.join(user_dir, ".cbsurge", "config.json")
        return config_file_path


    def get_config(self):
        """
        get config from ~/.cbsurge/config.json

        Returns:
            JSON object
        """
        config_file_path = self.get_config_file_path()
        if os.path.exists(config_file_path):
            with open(config_file_path, "r", encoding="utf-8") as data:
                return json.load(data)
        else:
            return None

    def get_config_value_by_key(self, key: str, default=None):
        """
        get config value by key

        Parameters:
            key (str): key
            default (str): default value if not exists. Default is None
        """
        if self.config is None:
            self.config = self.get_config()
        if self.config is not None:
            return self.config.get(key, default)
        else:
            return default


    def set_config_value_by_key(self, key: str, value):
        if self.config is None:
            self.config = {}
        self.config[key] = value


    def set_account_name(self, account_name: str):
        self.set_config_value_by_key("account_name", account_name)

    def get_account_name(self):
        return self.get_config_value_by_key("account_name")

    def set_stac_container_name(self, container_name: str):
        self.set_config_value_by_key("stac_container_name", container_name)

    def get_stac_container_name(self):
        return self.get_config_value_by_key("stac_container_name")

    def set_publish_container_name(self, container_name: str):
        self.set_config_value_by_key("publish_container_name", container_name)

    def get_publish_container_name(self):
        return self.get_config_value_by_key("publish_container_name")

    def set_file_share_name(self, file_share_name: str):
        self.set_config_value_by_key("file_share_name", file_share_name)

    def get_file_share_name(self):
        return self.get_config_value_by_key("file_share_name")

    def set_geohub_endpoint(self, account_name: str):
        self.set_config_value_by_key("geohub_endpoint", account_name)

    def get_geohub_endpoint(self):
        return self.get_config_value_by_key("geohub_endpoint")

    def save_config(self):
        """
        Save config.json under user directory as ~/.cbsurge/config.json
        """
        if self.get_account_name() is None:
            raise RuntimeError(f"account_name is not set")
        if self.get_stac_container_name() is None:
            raise RuntimeError(f"stac_container_name is not set")
        if self.get_file_share_name() is None:
            raise RuntimeError(f"file_share_name is not set")

        config_file_path = self.get_config_file_path()

        dir_path = os.path.dirname(config_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(config_file_path, "w", encoding="utf-8") as file:
            json.dump(self.config, file, ensure_ascii=False, indent=4)

        logger.debug(f"config file was saved to {config_file_path}")


    def get_credential(self, interactive_browser=False):
        """
        get token credential for azure.

        Usage example:

        from azure.storage.blob import BlobServiceClient
        from cbsurge.session import Session

        session = Session()
        credential = session.get_credential()

        blob_service_client = BlobServiceClient(
            account_url="https://<my_account_name>.blob.core.windows.net",
            credential=token_credential
        )

        Returns:
            Azure TokenCredential is returned if authenticated.
        """
        credential = SurgeTokenCredential()
        return credential


    def get_token(self, scopes = "https://storage.azure.com/.default"):
        """
        get access token for blob storage account. This token is required for using Azure REST API.

        Parameters:
            scopes: scopes for get_token method. Default to "https://storage.azure.com/.default"
        Returns:
            Azure token is returned if authenticated.
        Raises:
            ClientAuthenticationError is raised if authentication failed.

            ClientAuthenticationError:
            https://learn.microsoft.com/en-us/python/api/azure-core/azure.core.exceptions.clientauthenticationerror?view=azure-python
        """
        try:
            credential = self.get_credential()
            token = credential.get_token(scopes)
            return token
        except ClientAuthenticationError as err:
            logger.error("authentication failed. Please use 'rapida init' command to setup credentials.")
            raise err


    def authenticate(self, scopes = "https://storage.azure.com/.default"):
        """
        Authenticate to Azure

        Parameters:
            scopes: scopes for get_token method. Default to "https://storage.azure.com/.default"
        Returns:
            Azure credential and token are returned if authenticated. If authentication failed, return None.
        """
        try:
            credential = self.get_credential(interactive_browser=True)
            token = credential.get_token(scopes)
            return [credential, token]
        except ClientAuthenticationError as err:
            logger.error("authentication failed.")
            return None


    def get_blob_service_client(self, account_name: str = None) -> BlobServiceClient:
        """
        get BlobServiceClient for account url

        If the parameter is not set, use default account name from config.

        Usage example:
            async with Session() as session:
                async with session.get_blob_service_client(account_name="undpgeohub") as blob_service_client:
                    # do something

        Parameters:
            account_name (str): name of storage account. https://{account_name}.blob.core.windows.net
        Returns:
            BlobServiceClient
        """
        credential = self.get_credential()
        account_url = self.get_blob_service_account_url(account_name)
        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=credential
        )
        return blob_service_client

    def get_blob_container_client(self, account_name: str = None, container_name: str = None) -> ContainerClient:
        """
        get ContainerClient for account name and container name

        If the parameter is not set, use default account name from config.

        Usage example:
        async with Session() as session:
            async with session.get_blob_container_client() as container_client:
                # do something

        Parameters:
            account_name (str): name of storage account. https://{account_name}.blob.core.windows.net
            container_name (str): name of storage container name. https://{account_name}.blob.core.windows.net/{container_name}
        Returns:
            ContainerClient
        """
        credential = self.get_credential()
        account_url = self.get_blob_service_account_url(account_name)
        ct_name = container_name if container_name is not None else self.get_stac_container_name()
        container_client = ContainerClient(
            account_url=account_url,
            credential=credential,
            container_name=ct_name
        )
        return container_client

    def get_blob_service_account_url(self, account_name: str = None) -> str:
        """
        get blob service account URL

        If the parameter is not set, use default account name from config.

        Parameters:
            account_name (str): Optional. name of storage account url.
        """
        ac_name = account_name if account_name is not None else self.get_account_name()
        return f"https://{ac_name}.blob.core.windows.net"

    def get_file_share_account_url(self, account_name: str = None) -> str:
        """
        get blob service account URL

        If the parameter is not set, use default account name from config.

        Parameters:
            account_name (str): Optional. name of storage account url. If the parameter is not set, use default account name from config.
        """
        ac_name = account_name if account_name is not None else self.get_account_name()
        return f"https://{ac_name}.file.core.windows.net"

    def get_components(self):
        """
        Gets the available components in the config file
        :return: iterator[str]
        """
        variables = self.get_config_value_by_key(key='variables')
        return set(variables.keys())

    def get_component(self, component:str = None):
        """
        Get the config dict for a component
        :param component: str, name of the component
        :return: dict with its config extracted from the config  file
        """
        variables_elem = self.get_config_value_by_key(key='variables')
        return variables_elem[component]

    def get_variables(self, component: str = None):
        """
        Gets the config for a given variable from a component
        :param component:
        :param variable:
        :return:
        """
        component = self.get_component(component=component)
        return set(component.keys())

    def get_variable(self, component:str= None, variable=None ):
        """
        Gets the config for a given variable from a component
        :param component:
        :param variable:
        :return:
        """
        component = self.get_component(component=component)
        return component[variable]


def is_rapida_initialized():
    """
    Check if RAPIDA has been initialized

    :return: boolean if True, config.json exists
    """
    with Session() as session:
        if session.get_config() is None:
            logger.warning(f"Rapida tool is not initialized. Please run `rapida init` first.")
            return False
    return True

# for testing
# import asyncio
# async def main():
#     # Create the session object
#     async with Session() as session:
#         async with session.get_blob_container_client(container_name="stacdata") as container_client:
#             blob_name = "worldpop/2020/ABW/aggregate/ABW_active_total.tif"
#             print(f"Downloading blob: {blob_name}")
#             stream = await container_client.download_blob(blob=blob_name)
#             data = await stream.readall()
#             await container_client.close()
#
#             # Save the blob data to a local file
#             output_file = "ABW_active_total.tif"
#             with open(output_file, "wb") as file:
#                 file.write(data)
#
#             print(f"Blob downloaded successfully and saved as '{output_file}'")
#
# if __name__ == "__main__":
#     asyncio.run(main())
