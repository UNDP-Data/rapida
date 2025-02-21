
from azure.storage.blob import BlobServiceClient
from msal import PublicClientApplication, SerializableTokenCache
from azure.core.credentials import AccessToken, TokenCredential
from datetime import datetime, timedelta, UTC
import os
from os.path import expanduser
import logging
from playwright.sync_api import sync_playwright
import click


logger = logging.getLogger(__name__)

# cache file path
TOKEN_CACHE_FIlE = 'token_cache.json'

class PlaywrightAuthenticator():

    def  authenticate(self, url=None, auth_code=None):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Use headless=True for invisible mode
            page = browser.new_page()
            page.goto(url=url)
            page.wait_for_url(url)
            e = page.get_by_label('Code')
            print(e)
            browser.close()


class MsalTokenCredential(TokenCredential):

    def __init__(self, device_auth=True, cache_dir=None):
        # Azure AD Configuration
        # UNDP tenant ID
        self.tenant_id = os.environ['TENANT_ID']
        # public client id for AZURE CLI available from:
        # https://learn.microsoft.com/en-us/troubleshoot/entra/entra-id/governance/verify-first-party-apps-sign-in#application-ids-of-commonly-used-microsoft-applications
        self.client_id = os.environ['CLIENT_ID']
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"  #

        # cache file path
        if cache_dir is None:
            cache_dir = f"{expanduser("~")}/.cbsurge/"

        self._cache_file_ = f"{cache_dir}{TOKEN_CACHE_FIlE}"

        self._load_cache_()
        self.device_auth = device_auth
        self.app = PublicClientApplication(client_id=self.client_id , authority=self.authority, token_cache=self._cache_)

    def _load_cache_(self):
        self._cache_ = SerializableTokenCache()  # see https://github.com/AzureAD/microsoft-authentication-extensions-for-python

        dir_name = os.path.dirname(self._cache_file_)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        logger.debug(self._cache_file_)
        if os.path.exists(self._cache_file_):
            with open(self._cache_file_, "r") as cache_src:
                self._cache_.deserialize(cache_src.read())
    def _save_cache_(self):
        with open(self._cache_file_, "w") as cache_dst:
            cache_dst.write(self._cache_.serialize())

    def get_token(self, *scopes, **kwargs):
        scopes = list(scopes)
        accounts = self.app.get_accounts()
        result = self.app.acquire_token_silent(scopes, account=accounts[0] if accounts else None)
        if not result:
            logger.info("Token not found in cache. Initiating interactive login...")
            """
            FOR DOCKER, or an env without browser
            """
            if self.device_auth:
                flow = self.app.initiate_device_flow(scopes=scopes)

                if "user_code" not in flow:
                    raise ValueError("Device flow initialization failed. Check your configuration.")

                uri = flow['verification_uri']
                code = flow['user_code']
                logger.info(f"Visit {uri} and enter the code: {code}")
                # dd = dict(grant_type='urn:ietf:params:oauth:grant-type:device_code', client_id = CLIENT_ID,device_code=code)
                # r = httpx.post(f'https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token', data=dd)
                #
                # r.raise_for_status()
                # print(r.status_code)
                # authenticator = PlaywrightAuthenticator()
                # result = authenticator.authenticate(url=uri, auth_code=code)

                result = self.app.acquire_token_by_device_flow(flow)
            else:
                result = self.app.acquire_token_interactive(scopes=scopes)
            self._save_cache_()

        current_time = datetime.now(UTC)
        expires_in = timedelta(seconds=result["expires_in"])
        expires_on = current_time + expires_in
        return AccessToken(result["access_token"], int(expires_on.timestamp()))


@click.command()
@click.option('-c', '--cache_dir', default=None, type=click.Path(),
              help="Optional. Folder path to store token_cache.json. Default is ~/.cbsurge folder to store cache file.")
def authenticate(cache_dir):
    """
    Authenticate with MSAL locally to UNDP account
    """
    if cache_dir is None:
        cache_dir = f"{expanduser("~")}/.cbsurge/"
    credential = MsalTokenCredential(device_auth=False, cache_dir=cache_dir)

    # Connect to Azure Blob Storage
    blob_service_client = BlobServiceClient(
        account_url="https://undpgeohub.blob.core.windows.net",
        credential=credential
    )

    # List containers in the storage account
    cc = blob_service_client.get_container_client(container='stacdata')
    for blob in cc.list_blobs(name_starts_with='worldpop/2020/ABW'):
        blob_client = cc.get_blob_client(blob)
        print(blob_client.blob_name)


if __name__ == '__main__':
    authenticate()