
from azure.storage.blob import BlobServiceClient
import time
from urllib.parse import urlparse, parse_qs
from msal import PublicClientApplication, SerializableTokenCache
from azure.core.credentials import AccessToken, TokenCredential
from datetime import datetime, timedelta, UTC
import os
import hashlib
import base64
from os.path import expanduser
import logging
from playwright.sync_api import sync_playwright
from cbsurge.util.in_notebook import in_notebook
from requests_oauthlib import OAuth2Session
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import json
logger = logging.getLogger(__name__)

# Secure AES-256 key (store in a vault or HSM in production)
KEY = os.urandom(32)  # WARNING: Don't regenerate every time! Use a fixed key.

# Secure RAM-based storage location
SHM_DIR = "/dev/shm/cbcurge"

# Ensure secure directory exists (only accessible by the owner)
os.makedirs(SHM_DIR, mode=0o700, exist_ok=True)  # Owner-only access


def derive_key_from_username(username: str) -> bytes:
    """Derives a 256-bit AES key from the username using SHA-256."""
    # Hash the username using SHA-256 to create a fixed-length key (32 bytes)
    key = hashlib.sha256(username.encode('utf-8')).digest()
    return key


def encrypt_json(json_data: dict, username: str, shm_path: str = None):
    """Encrypt a JSON object and store it securely in /dev/shm/ as a binary file."""
    # Derive AES key from username
    key = derive_key_from_username(username)

    # Initialize AES-GCM with the derived key
    aesgcm = AESGCM(key)

    # Generate a 96-bit (12-byte) nonce for AES-GCM
    nonce = os.urandom(12)

    # Convert JSON data to bytes
    plaintext = json.dumps(json_data).encode('utf-8')

    # Encrypt the plaintext using AES-GCM
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    # Write the nonce and ciphertext to the specified file path
    with open(shm_path, "wb") as f:
        f.write(nonce + ciphertext)

    # Ensure the file exists and restrict access to owner-only
    assert os.path.exists(shm_path)
    os.chmod(shm_path, 0o600)


def decrypt_json(username: str, shm_path: str) -> dict:
    """Decrypt a JSON object stored in a binary file and return it as a Python dictionary."""
    # Derive AES key from username
    key = derive_key_from_username(username)

    # Read the encrypted file and extract the nonce and ciphertext
    with open(shm_path, "rb") as f:
        data = f.read()

    # First 12 bytes are the nonce
    nonce = data[:12]

    # Remaining bytes are the ciphertext
    ciphertext = data[12:]

    # Initialize AES-GCM with the derived key
    aesgcm = AESGCM(key)

    # Decrypt the ciphertext
    decrypted_data = aesgcm.decrypt(nonce, ciphertext, None)

    # Convert the decrypted bytes back to a Python dictionary (JSON)
    json_data = json.loads(decrypted_data.decode('utf-8'))

    return json_data
class SurgeTokenCredential(TokenCredential):

    KEY = derive_key_from_username(os.environ.get('USER', None))
    TOKEN_FILE_NAME = f'{base64.urlsafe_b64encode(KEY).decode('utf-8')[:25]}.bin'
    def __init__(self, cache_dir=SHM_DIR):
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
        if cache_dir is None:
            cache_dir = os.path.join(expanduser("~"),'.cbsurge')
        self._cache_file_ = os.path.join(cache_dir,self.TOKEN_FILE_NAME)

        # if os.path.exists(self._cache_file_):
        #     os.remove(self._cache_file_)
        self._load_cache_()

        #self.app = PublicClientApplication(client_id=self.client_id , authority=self.authority, token_cache=self._cache_)

    def _load_cache_(self):
        if os.path.exists(self._cache_file_):
            self.token = decrypt_json(username=os.environ.get('USER', None), shm_path=self._cache_file_)
    def _save_cache_(self):
        encrypt_json(json_data=self.token, username=os.environ.get('USER', None), shm_path=self._cache_file_)

    def fetch_token(self, *scope, email=None, passwd=None, **kwargs):
        # Start OAuth session
        oauth = OAuth2Session(self.client_id, redirect_uri=self.redirect_uri, scope=scope)
        # Step 1: Get authorization URL
        auth_url, state = oauth.authorization_url(self.auth_url)
        response = requests.get(auth_url, allow_redirects=False)
        auth_code = None
        with sync_playwright() as p:
            for i in range(3):
                try:
                    browser = p.chromium.launch(headless=True,
                                                args=["--no-sandbox"])  # Use headless=True for invisible mode
                    page = browser.new_page()
                    page.goto(url=auth_url)
                    # Fill in the username
                    page.fill(selector='input[type="email"]', value=email)
                    page.click('input[type="submit"]')  # Click the next button

                    # Wait for the password field to appear and fill it in
                    page.wait_for_selector(selector='input[type="password"]', timeout=.5 * 60 * 1000)
                    page.fill(selector='input[type="password"]', value=passwd)
                    page.click('input[type="submit"]')  # Click the sign-in button
                    # # Wait for the element containing the number to appear
                    page.wait_for_selector(selector='#idRichContext_DisplaySign', timeout=.5 * 60 * 1000)
                    ds = page.locator("#idRichContext_DisplaySign")

                    number = int(ds.text_content())

                    print(f"Enter {number} into Authenticator App")
                    # Wait for the redirection to complete
                    page.wait_for_url("**/*code=*",
                                      timeout=.5 * 60 * 1000)  # Adjust the pattern to match the expected redirect URL

                    # Get the current URL after redirection
                    final_url = page.url
                    parsed_url = urlparse(final_url)
                    query_params = parse_qs(parsed_url.query)
                    auth_code = query_params.get("code", [None])[0]
                except Exception as e:
                    print(e)



            # Close the browser

            browser.close()
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
        token_data['expires_in'] = int(token_data['expires_in'] + time.time())
        token_data['cached_at'] = int(time.time())
        return token_data

    def refresh_token(self):
        data = {
            'client_id': self.client_id,
            'grant_type': 'refresh_token',
            'refresh_token': self.token['refresh_token'],
            'scope': self.token['scope']
        }

        response = requests.post(self.token_url, data=data)

        if response.status_code == 200:
            token_data = response.json()
            token_data['expires_in'] = int(token_data['expires_in'] + time.time())
            token_data['cached_at'] = int(time.time())
            return token_data
        else:
            logger.info("Failed to refresh token:", response.json())


    def get_token(self, *scopes, mfa_widget=None) -> AccessToken:
        'get the access token, aither from cache, fetched or refreshed'
        if self.token:
            expires_at = datetime.fromtimestamp(self.token["expires_in"])
            expires_in = expires_at - datetime.now()
            expires_in_secs = expires_in.total_seconds()

            if expires_in_secs < 15*60:
                logger.debug(f'Token cached at {self._cache_file_} will expire in {expires_in_secs} secs and is going ot be refreshed')
                self.token = self.refresh_token()
                self._save_cache_()

        else :
            logger.info("Token not found in cache. Acquiring...")
            """
            FOR DOCKER, or an env without browser
            """
            self.token = self.fetch_token(*scopes, mfa_widget=mfa_widget)
            self._save_cache_()

        return AccessToken(self.token["access_token"], self.token['expires_in'])