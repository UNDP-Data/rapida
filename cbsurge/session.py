import logging
import click
from azure.identity import DefaultAzureCredential, AzureAuthorityHosts
from azure.core.exceptions import ClientAuthenticationError
from azure.storage.blob import BlobServiceClient


logger = logging.getLogger(__name__)


class Session(object):
    def __init__(self, scopes = "https://storage.azure.com/.default"):
        """
        constructor

        Parameters:
            scopes: scopes for get_token method. Default to "https://storage.azure.com/.default"
        """
        self.scopes = scopes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.scopes = None


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

    def get_token(self):
        """
        get access token for blob storage account. This token is required for using Azure REST API.

        Returns:
            Azure token is returned if authenticated.
        Raises:
            ClientAuthenticationError is raised if authentication failed.

            ClientAuthenticationError:
            https://learn.microsoft.com/en-us/python/api/azure-core/azure.core.exceptions.clientauthenticationerror?view=azure-python
        """
        try:
            credential = self.get_credential()
            token = credential.get_token(self.scopes)
            return token
        except ClientAuthenticationError as err:
            logger.error("authentication failed. Please use 'rapida init' command to setup credentials.")
            raise err

    def authenticate(self):
        """
        Authenticate to Azure through interactive browser if DefaultAzureCredential is not provideds.
        Authentication uses DefaultAzureCredential.

        Please refer to https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python
        about DefaultAzureCredential api specificaiton.

        Returns:
            Azure credential and token are returned if authenticated. If authentication failed, return None.
        """
        try:
            credential = DefaultAzureCredential(
                exclude_interactive_browser_credential=False,
            )
            token = credential.get_token(self.scopes)
            return [credential, token]
        except ClientAuthenticationError as err:
            logger.error("authentication failed.")
            logger.error(err)
            return None

    def get_blob_service_client(self, account_name: str) -> BlobServiceClient:
        """
        get BlobServiceClient for account url

        Usage example:
            with Session() as session:
                blob_service_client = session.get_blob_service_client(
                    account_name="undpgeohub"
                )

        Parameters:
            account_name (str): name of storage account. https://{account_name}.blob.core.windows.net
        Returns:
            BlobServiceClient
        """
        credential = self.get_credential()
        blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=credential
        )
        return blob_service_client


@click.command()
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug"
              )
def init(debug=False):
    """
    This command setup rapida command environment by authenticating to Azure.
    """
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, force=True)

    if click.confirm('Would you like to setup rapida tool?', abort=True):
        # login to Azure
        session = Session()
        credential, token = session.authenticate()
        logger.debug(token)
        if token is None:
            logger.info("Authentication failed.Please `az login` to authenticate first.")
            return
        click.echo('Setting up was successfully done!')

