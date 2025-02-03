import logging
import os
import json
import click
from azure.identity import DefaultAzureCredential, AzureAuthorityHosts
from azure.core.exceptions import ClientAuthenticationError
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from azure.storage.fileshare.aio import ShareServiceClient
from osgeo import gdal
from collections import UserDict
import shutil

logger = logging.getLogger(__name__)
gdal.UseExceptions()

class Config(UserDict):
    def __init__(self, file_path, **kwargs):
        """
        Initialize the AutoSaveDict with a target JSON filename.
        If the file exists, load its contents; otherwise, start with an empty dict
        (or with any provided initial values).

        :param file_path: Path to the JSON file where the dictionary will be saved.
        :param args: Positional arguments passed to dict().
        :param kwargs: Keyword arguments passed to dict().
        """
        self.file_path = file_path

        # If the file exists, load its contents
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    loaded_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load JSON from {self.file_path}: {e}")
                loaded_data = {}
        else:
            loaded_data = {}

        # Merge with any additional data provided during instantiation.
        # Values provided in args/kwargs will override those loaded from the file.
        loaded_data.update(dict(**kwargs))

        # Initialize the underlying dictionary with the merged data.
        super().__init__(**loaded_data)
        self._save()  # Write the initial state (in case the file didn't exist)

    def __setitem__(self, key, value):
        """Set the item and save the dictionary to the JSON file."""
        super().__setitem__(key, value)
        self._save()

    def __delitem__(self, key):
        """Delete the item and save the dictionary to the JSON file."""
        super().__delitem__(key)
        self._save()

    def update(self, *args, **kwargs):
        """Update the dictionary and save the changes."""
        super().update( *args, **kwargs)
        self._save()

    def clear(self):
        """Clear the dictionary and save the changes."""
        super().clear()
        self._save()

    def pop(self, key, *args):
        """Remove the specified key and save the dictionary."""
        value = super().pop(key, *args)
        self._save()
        return value

    def popitem(self):
        """Remove and return an arbitrary (key, value) pair and save the dictionary."""
        item = super().popitem()
        self._save()
        return item

    def _save(self):
        """Write the current state of the dictionary to the JSON file."""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            print(f"Error saving AutoSaveDict to {self.file_path}: {e}")


class Project:
    config_file_name = 'rapida.json'
    data_folder = None
    geopackage_file:str = None
    name:str = None
    def __init__(self, folder:str=None, polygons:str=None, mask:str=None, projection='ESRI:54009' ):

        folder = os.path.abspath(folder)
        config_file = os.path.join(folder, self.config_file_name)
        if os.path.exists(config_file):
            self.config = Config(file_path=config_file)
            for k, v in self.config.items():
                self.__setattr__(k, v)

        else:
            self.folder = folder
            *rest, name = self.folder.split(os.path.sep)
            self.name = name
            self.config_file = os.path.join(self.folder, self.config_file_name)
            self.data_folder = os.path.join(self.folder, 'data')
            self.geopackage_file = os.path.join(self.data_folder, f'{self.name}.gpkg')
            if not os.path.exists(self.folder):
                logger.debug(f'Creating project folder ...')
                os.makedirs(self.folder)
                self.config = Config(file_path=self.config_file,
                                     name=self.name,
                                     config_file=self.config_file,
                                     folder=self.folder,

                                     )
                logger.debug(f'Creating project config file ...')
                logger.info(f'Project {self.name} was created in {self.folder}')
            else:
                if os.path.exists(self.config_file):
                    self.config = Config(file_path=self.config_file)

                else:
                    logger.info(f'Creating project config file ...')
                    self.config = Config(file_path=self.config_file,
                                         name=self.name,
                                         config_file=self.config_file,
                                         folder=self.folder,
                                         )

                    logger.info(f'Project {self.name} is located in {self.folder}')

        if polygons is not None:
            with gdal.OpenEx(polygons) as poly_ds:
                lcount = poly_ds.GetLayerCount()
                if lcount > 1:
                    lnames = list()
                    for i in range(lcount):
                        l = poly_ds.GetLayer(i)
                        lnames.append(l.GetName())
                    #click.echo(f'{polygons} contains {lcount} layers: {",".join(lnames)}')
                    layer_name = click.prompt(
                        f'{polygons} contains {lcount} layers: {",".join(lnames)} Please type/select  one or pres enter to skip if you wish to use default value',
                        type=str, default=lnames[0])
                else:
                    layer_name = poly_ds.GetLayer(0).GetName()
                if not os.path.exists(self.data_folder):
                    os.makedirs(self.data_folder)
                gdal.VectorTranslate(self.geopackage_file, poly_ds, format='GPKG',reproject=True, dstSRS=projection,
                                 layers=[layer_name], layerName='polygons', geometryType='PROMOTE_TO_MULTI', makeValid=True)
        # if mask is not  None:
        #     try:
        #         vm_ds = gdal.OpenEx(mask, gdal.OF_VECTOR)
        #     except RuntimeError as ioe:
        #         if 'supported' in str(ioe):
        #             vm_ds = None
        #         else:
        #             raise
        #     if vm_ds is not None:
        #         pass
        #
        # assert self.is_valid, f'{self} is not valid'




    def __str__(self):
        txt =   f'''
    name:            {self.name}
    folder:          {self.folder}
    config file:     {self.config_file}
    data folder:     {self.data_folder}
    geopackage file: {self.geopackage_file}
                '''
        return txt

    @property
    def is_valid(self):
        assert os.path.exists(self.folder), f'{self.folder} does not exist'
        can_write = os.access(self.folder, os.W_OK)
        proj_cfg_file_exists = os.path.exists(self.config_file)
        proj_cfg_file_is_empty = os.path.getsize(self.config_file) == 0
        geopackage_file_path_is_defined = self.geopackage_file not in (None, '')
        return can_write and proj_cfg_file_exists and not proj_cfg_file_is_empty



    def delete(self):
        if click.confirm(f'Are you sure you want to delete {self.name} located in {self.folder} ?', abort=True):
            shutil.rmtree(self.folder)


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
        if self.config is not None:
            logger.debug(f"rapida config was loaded")


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
    @property
    def project(self):
        project_folder = self.config.get('project', None)
        if project_folder is None:
            raise KeyError(f'No Rapida project has been defined yet. Consider creating one first ')
        prj = Project(folder=project_folder)
        return prj

    def set_config_value_by_key(self, key: str, value):
        if self.config is None:
            self.config = {}
        self.config[key] = value


    def set_root_data_folder(self, folder_name):
        self.set_config_value_by_key("root_data_folder", folder_name)

    def get_root_data_folder(self, is_absolute_path=True):
        """
        get root data folder

        Parameters:
            is_absolute_path (bool): Optional. If true, return absolute path, otherwise relative path. Default is True.
        Returns:
            root data folder path (str)
        """
        root_data_folder = self.get_config_value_by_key("root_data_folder")
        if is_absolute_path:
            return  os.path.expanduser(root_data_folder)
        else:
            return root_data_folder

    def set_account_name(self, account_name: str):
        self.set_config_value_by_key("account_name", account_name)

    def get_account_name(self):
        return self.get_config_value_by_key("account_name")

    def set_container_name(self, container_name: str):
        self.set_config_value_by_key("container_name", container_name)

    def get_container_name(self):
        return self.get_config_value_by_key("container_name")

    def set_file_share_name(self, file_share_name: str):
        self.set_config_value_by_key("file_share_name", file_share_name)

    def get_file_share_name(self):
        return self.get_config_value_by_key("file_share_name")

    def save_config(self):
        """
        Save config.json under user directory as ~/.cbsurge/config.json
        """
        if self.get_root_data_folder() is None:
            raise RuntimeError(f"root_data_folder is not set")
        if self.get_account_name() is None:
            raise RuntimeError(f"account_name is not set")
        if self.get_container_name() is None:
            raise RuntimeError(f"container_name is not set")
        if self.get_file_share_name() is None:
            raise RuntimeError(f"file_share_name is not set")

        config_file_path = self.get_config_file_path()

        dir_path = os.path.dirname(config_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(config_file_path, "w", encoding="utf-8") as file:
            json.dump(self.config, file, ensure_ascii=False, indent=4)

        logger.debug(f"config file was saved to {config_file_path}")


    def get_credential(self):
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
        credential = DefaultAzureCredential()

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
        Authenticate to Azure through interactive browser if DefaultAzureCredential is not provideds.
        Authentication uses DefaultAzureCredential.

        Please refer to https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python
        about DefaultAzureCredential api specificaiton.

        Parameters:
            scopes: scopes for get_token method. Default to "https://storage.azure.com/.default"
        Returns:
            Azure credential and token are returned if authenticated. If authentication failed, return None.
        """
        try:
            credential = DefaultAzureCredential(
                exclude_interactive_browser_credential=False,
            )
            token = credential.get_token(scopes)
            return [credential, token]
        except ClientAuthenticationError as err:
            logger.error("authentication failed.")
            logger.error(err)
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
        ct_name = container_name if container_name is not None else self.get_container_name()
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

    def get_share_service_client(self, account_name: str = None, share_name: str = None) -> ShareServiceClient:
        """
        get ShareServiceClient for account url

        If the parameter is not set, use default account name from config.

        Usage example:
            with Session() as session:
                share_service_client = session.get_share_service_client(
                    account_name="undpgeohub",
                    share_name="cbrapida"
                )

        Parameters:
            account_name (str): name of storage account.
            share_name (str): name of file share.

            both parameters are equivalent to the below URL's bracket places.

            https://{account_name}.file.core.windows.net/{share_name}
        Returns:
            ShareServiceClient
        """
        credential = self.get_credential()
        account_url = self.get_file_share_account_url(account_name, share_name)
        share_service_client = ShareServiceClient(
            account_url=account_url,
            credential=credential
        )
        return share_service_client

    def get_file_share_account_url(self, account_name: str = None, share_name: str = None) -> str:
        """
        get blob service account URL

        If the parameter is not set, use default account name from config.

        Parameters:
            account_name (str): Optional. name of storage account url. If the parameter is not set, use default account name from config.
            share_name (str): name of file share. If the parameter is not set, use default account name from config.
        """
        ac_name = account_name if account_name is not None else self.get_account_name()
        sh_name = share_name if share_name is not None else self.get_file_share_name()
        return f"https://{ac_name}.file.core.windows.net/{sh_name}"

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
        variables = self.get_config_value_by_key(key='variables')
        return variables[component]

    def get_variable(self, component:str= None, variable=None ):
        """
        Gets the config for a given variable from a component
        :param component:
        :param variable:
        :return:
        """
        component = self.get_component(component=component)
        return component[variable]
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
