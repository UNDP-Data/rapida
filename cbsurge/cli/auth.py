import logging
from cbsurge.az.surgeauth import SurgeTokenCredential
import click


logger = logging.getLogger(__name__)

# cache file path
TOKEN_CACHE_FIlE = 'token_cache.json'


@click.command()
@click.option('-c', '--cache_dir', default=None, type=click.Path(),
              help="Optional. Folder path to store token_cache.json. Default is ~/.cbsurge folder to store cache file.")

def authenticate(cache_dir):
    """
    Authenticate with  UNDP account
    """
    credential = SurgeTokenCredential(cache_dir=cache_dir)
    token, exp_ts = credential.get_token("https://storage.azure.com/.default",)


if __name__ == '__main__':
    authenticate()