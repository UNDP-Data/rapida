import click
import httpx
import time
from urllib.parse import urlparse, parse_qs
from azure.core.credentials import AccessToken, TokenCredential
from datetime import datetime
import os
import hashlib
import base64
from os.path import expanduser
from cbsurge.util.setup_logger import setup_logger
from oauthlib.oauth2 import OAuth2Error
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from requests_oauthlib import OAuth2Session
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import json

logger = setup_logger(name='rapida', make_root=False)


def derive_key_from_username(username: str) -> bytes:
    """Derives a 256-bit AES key from the username using SHA-256."""
    # Hash the username using SHA-256 to create a fixed-length key (32 bytes)
    key = hashlib.sha256(username.encode('utf-8')).digest()
    return key


def encrypt_json(json_data: dict, username: str, cache_file_path: str = None):
    """Encrypt a JSON object and store it securely as a binary file."""
    # Derive AES key from username
    key = derive_key_from_username(username)

    # Initialize AES-GCM with the derived key
    aes_gcm = AESGCM(key)

    # Generate a 96-bit (12-byte) nonce for AES-GCM
    nonce = os.urandom(12)

    # Convert JSON data to bytes
    plaintext = json.dumps(json_data).encode('utf-8')

    # Encrypt the plaintext using AES-GCM
    ciphertext = aes_gcm.encrypt(nonce, plaintext, None)

    # Write the nonce and ciphertext to the specified file path
    with open(cache_file_path, "wb") as f:
        f.write(nonce + ciphertext)

    # Ensure the file exists and restrict access to owner-only
    assert os.path.exists(cache_file_path)
    os.chmod(cache_file_path, 0o600)


def decrypt_json(username: str, cache_file_path: str) -> dict:
    """Decrypt a JSON object stored in a binary file and return it as a Python dictionary."""
    # Derive AES key from username
    key = derive_key_from_username(username)

    # Read the encrypted file and extract the nonce and ciphertext
    with open(cache_file_path, "rb") as f:
        data = f.read()

    # First 12 bytes are the nonce
    nonce = data[:12]

    # Remaining bytes are the ciphertext
    ciphertext = data[12:]

    # Initialize AES-GCM with the derived key
    aes_gcm = AESGCM(key)

    # Decrypt the ciphertext
    decrypted_data = aes_gcm.decrypt(nonce, ciphertext, None)

    # Convert the decrypted bytes back to a Python dictionary (JSON)
    json_data = json.loads(decrypted_data.decode('utf-8'))

    return json_data




def is_called_from_click():
    try:
        return click.get_current_context(silent=True) is not None
    except RuntimeError:
        return False  # Happens if Click is not in use



class SurgeTokenCredential(TokenCredential):

    KEY = derive_key_from_username(os.environ.get('USER', None))
    TOKEN_FILE_NAME = f'{base64.urlsafe_b64encode(KEY).decode('utf-8')[:25]}.bin'
    STORAGE_SCOPE = ["https://storage.azure.com/.default"]

    def __init__(self, cache_dir=None):
        # Azure AD Configuration
        # UNDP tenant ID
        self.token = None
        self.tenant_id = os.environ['TENANT_ID']
        # public client id for AZURE CLI available from:
        # https://learn.microsoft.com/en-us/troubleshoot/entra/entra-id/governance/verify-first-party-apps-sign-in#application-ids-of-commonly-used-microsoft-applications
        self.client_id = os.environ['CLIENT_ID']
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.redirect_uri = "https://login.microsoftonline.com/common/oauth2/nativeclient"
        self.auth_url = f"{self.authority}/oauth2/v2.0/authorize"
        self.token_url = f"{self.authority}/oauth2/v2.0/token"

        # cache file path
        self._cache_dir_ = cache_dir
        if self._cache_dir_ is None:
            self._cache_dir_ = os.path.join(expanduser("~"), '.cbsurge')
        self._cache_file_ = os.path.join(self._cache_dir_,self.TOKEN_FILE_NAME)
        self._load_from_cache_()


    def _load_from_cache_(self):
        if os.path.exists(self._cache_file_):
            self.token = decrypt_json(username=os.environ.get('USER', None), cache_file_path=self._cache_file_)

    def _save_to_cache_(self):
        if not os.path.exists(self._cache_dir_):
            os.makedirs(self._cache_dir_, mode=0o700, exist_ok=True)
        encrypt_json(json_data=self.token, username=os.environ.get('USER', None), cache_file_path=self._cache_file_)

    async def fetch_token_async(self, *scope, username=None, password=None, mfa_confirmation_widget=None):
        # Start OAuth session

        oauth = OAuth2Session(self.client_id, redirect_uri=self.redirect_uri, scope=scope)

        # Step 1: Get authorization URL
        auth_url, state = oauth.authorization_url(self.auth_url)

        auth_code = None

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])  # Launch headless browser
            page = await browser.new_page()
            error_msg = None
            await page.goto(auth_url)


            # Wait for the email field to appear and fill it in
            await page.wait_for_selector('input[type="email"]', timeout=30 * 1000)
            await page.fill('input[type="email"]', value=str(username))
            await page.click('input[type="submit"]')  # Click the next button


            # Error handling for username
            error_selector = '#usernameError'  # Match the ID of the error message element
            try:
                await page.wait_for_selector(error_selector, timeout=3000)  # Wait for error message
                error_message = await page.inner_text(error_selector)  # Capture the error message
                error_msg = f'Failed to authenticate using username "{username}". {error_message}'
            except Exception:
                pass  # Ignore error if the selector is not found

            if error_msg is not None:
                raise OAuth2Error(description=error_msg, status_code=None)

            # Wait for the password field to appear and fill it in
            await page.wait_for_selector('input[type="password"]', timeout=30 * 1000)
            await page.fill('input[type="password"]', value=str(password))
            await page.click('input[type="submit"]')  # Click the sign-in button

            # Error handling for password
            error_selector = '#passwordError'  # Match the ID of the error message element
            try:
                await page.wait_for_selector(error_selector, timeout=3000)  # Wait for error message
                error_msg = await page.inner_text(error_selector)  # Capture the error message
            except Exception:
                pass  # Ignore error if the selector is not found

            if error_msg is not None:
                raise OAuth2Error(description=error_msg)

            # Wait for the MFA element to appear
            await page.wait_for_selector('#idRichContext_DisplaySign', timeout=30 * 1000)
            # Get the MFA number
            number = await page.inner_text('#idRichContext_DisplaySign')
            mfa_msg = f"Your MFA input code is: {number}. Please input it into the Authenticator App on your mobile"
            logger.debug(mfa_msg)

            if mfa_confirmation_widget is not  None:
                mfa_confirmation_widget.value = f'<b>Your MFA input code is: <span style="color:red">{number}</span>. Please input it into the Authenticator App on your mobile</b>'

            # Wait for the redirection URL after MFA confirmation
            await page.wait_for_url("**/*code=*",
                                    timeout=30 * 1000)  # Adjust the pattern to match the expected redirect URL

            # Get the current URL after redirection
            final_url = page.url
            parsed_url = urlparse(final_url)
            query_params = parse_qs(parsed_url.query)
            auth_code = query_params.get("code", [None])[0]

            await browser.close()  # Close the browser after completion
        # Step 3: Exchange authorization code for access token
        token_data = {
            "client_id": self.client_id,
            "scope": "https://storage.azure.com/.default offline_access openid profile",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        token_response = requests.post(self.token_url, data=token_data)
        token_data = token_response.json()
        now = time.time()
        token_data['expires_at'] = int(token_data['expires_in'] + now)
        token_data['cached_at'] = int(now)
        return token_data



    def fetch_token_sync(self, *scope, username=None, password=None):

        # Start OAuth session
        oauth = OAuth2Session(self.client_id, redirect_uri=self.redirect_uri, scope=scope)
        # Step 1: Get authorization URL
        auth_url, state = oauth.authorization_url(self.auth_url)
        auth_code = None

        with sync_playwright() as p:
            with p.chromium.launch(headless=True, args=["--no-sandbox"]) as browser: # Use headless=True for invisible mode
                error_msg = None

                page = browser.new_page()
                page.goto(auth_url)
                # # Wait for the password field to appear and fill it in
                page.wait_for_selector(selector='input[type="email"]', timeout=30 * 1000)
                # Fill in the username
                page.fill(selector='input[type="email"]', value=str(username))
                page.click('input[type="submit"]')  # Click the next button

                error_selector = '#usernameError'  # This should match the ID of the error message element
                try:
                    page.wait_for_selector(selector=error_selector,timeout=3000)  # Wait up to 10 seconds for the error message
                    error_message = page.inner_text(error_selector)  # Capture the text of the error message
                    error_msg = f'Failed to authenticate using username "{username}". {error_message} '
                except Exception as e:
                    pass
                if error_msg is not None:
                    raise OAuth2Error(description=error_msg,status_code=None)

                # Wait for the password field to appear and fill it in
                page.wait_for_selector(selector='input[type="password"]', timeout=30 * 1000)
                page.fill(selector='input[type="password"]', value=str(password))
                page.click('input[type="submit"]')  # Click the sign-in button

                error_selector = '#passwordError'  # This should match the ID of the error message element
                try:
                    page.wait_for_selector(selector=error_selector,
                                           timeout=3000)  # Wait up to 10 seconds for the error message
                    error_msg = page.inner_text(error_selector)  # Capture the text of the error message
                except Exception as e:
                    pass
                if error_msg is not None:
                    raise OAuth2Error(description=error_msg)


                # Wait for the element containing the number to appear
                page.wait_for_selector(selector='#idRichContext_DisplaySign', timeout=30 * 1000)
                ds = page.locator("#idRichContext_DisplaySign")

                number = int(ds.text_content())

                logger.info(f"Your MFA input code is: {number}. Please input it into the  Authenticator App on your mobile")

                # Wait for the redirection to complete
                page.wait_for_url("**/*code=*",
                                  timeout=30 * 1000)  # Adjust the pattern to match the expected redirect URL

                # Get the current URL after redirection
                final_url = page.url
                parsed_url = urlparse(final_url)
                query_params = parse_qs(parsed_url.query)
                auth_code = query_params.get("code", [None])[0]


        # Step 3: Exchange authorization code for access token
        token_data = {
            "client_id": self.client_id,
            "scope": "https://storage.azure.com/.default offline_access openid profile",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        token_response = requests.post(self.token_url, data=token_data)
        token_data = token_response.json()
        now = time.time()
        token_data['expires_at'] = int(token_data['expires_in'] + now)
        token_data['cached_at'] = int(now)
        return token_data

    def refresh_token(self, *scope):
        data = {
            'client_id': self.client_id,
            'grant_type': 'refresh_token',
            'refresh_token': self.token['refresh_token'],
            'scope': list(scope)
        }


        response = requests.post(self.token_url, data=data)

        if response.status_code == 200:
            token_data = response.json()
            now = time.time()
            token_data['expires_at'] = int(token_data['expires_in'] + now)
            token_data['cached_at'] = int(now)
            return token_data
        else:

            logger.error(f"Failed to refresh token: {response.json()} " )


    async def refresh_token_async(self, *scope):

        data = {
            'client_id': self.client_id,
            'grant_type': 'refresh_token',
            'refresh_token': self.token['refresh_token'],
            'scope': list(scope)
        }

        # Use async HTTP request with httpx
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.token_url, data=data)

                if response.status_code == 200:
                    token_data = response.json()
                    now = time.time()
                    token_data['expires_at'] = int(token_data['expires_in'] + now)
                    token_data['cached_at'] = int(now)
                    return token_data
                else:
                    logger.error(f"Failed to refresh token: {response.json()} ")
            except httpx.RequestError as e:
                logger.error(f"An error occurred while requesting the token: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

    @property
    def authenticated(self):
        if self.token is not None:
            expires_at = datetime.fromtimestamp(self.token["expires_at"])
            expires_in = expires_at - datetime.now()
            expires_in_secs = expires_in.total_seconds()
            return expires_in_secs > 15
        return False

    async def get_token_async(self, *scopes, email=None, password=None, mfa_widget=None):

        if self.token:
            expires_at = datetime.fromtimestamp(self.token["expires_at"])
            expires_in = expires_at - datetime.now()
            expires_in_secs = expires_in.total_seconds()
            logger.info(
                f'Token cached at {self._cache_file_} will expire in {expires_in_secs} secs')
            if expires_in_secs < 15*60:
                logger.info(f'Refreshing token')
                new_token = await self.refresh_token_async(*scopes)
                if new_token is not None and 'access_token' in new_token:
                    self.token = new_token
                    self._save_to_cache_()
                else:
                    logger.debug(f"Attempting to authenticate with {self.auth_url}")
                    self.token = await self.fetch_token_async(*scopes, username=email, password=password,
                                                  mfa_confirmation_widget=mfa_widget)
                    self._save_to_cache_()

        else :
            logger.debug(f"Attempting to authenticate at {self.auth_url}")
            self.token = await self.fetch_token_async(*scopes, username=email, password=password,
                                                      mfa_confirmation_widget=mfa_widget)
            self._save_to_cache_()

        return AccessToken(self.token["access_token"], self.token['expires_at'])


    def get_token(self, *scopes) -> AccessToken:
        """
        get the access token, either from cache, fetched or refreshed
        :param scopes:

        :return:
        """


        if self.token:
            expires_at = datetime.fromtimestamp(self.token["expires_at"])
            expires_in = expires_at - datetime.now()
            expires_in_secs = expires_in.total_seconds()
            logger.info(
                f'Token cached at {self._cache_file_} will expire in {expires_in_secs} secs')
            if expires_in_secs < 15*60:
                logger.info(f'Refreshing token')
                new_token = self.refresh_token(*scopes) # 97hrs/
                if new_token is not None and 'access_token' in new_token:
                    self.token = new_token
                    self._save_to_cache_()
                else:
                    logger.info(f"Attempting to authenticate at {self.auth_url}")

                    email = os.environ.get('RAPIDA_USER', None) or click.prompt('Your UNDP email address please...',
                                                                                type=str)
                    password = os.environ.get('RAPIDA_PASSWORD', None) or click.prompt('Your password...', type=str,
                                                                                       hide_input=True)
                    self.token = self.fetch_token_sync(*scopes, username=email, password=password,)
                    self._save_to_cache_()

        else :
            logger.info(f"Attempting to authenticate at {self.auth_url}")
            email = os.environ.get('RAPIDA_USER', None) or click.prompt('Your UNDP email address please...', type=str)
            password = os.environ.get('RAPIDA_PASSWORD', None) or click.prompt('Your password...', type=str,
                                                                               hide_input=True)
            self.token = self.fetch_token_sync(*scopes, username=email, password=password)
            self._save_to_cache_()

        return AccessToken(self.token["access_token"], self.token['expires_at'])