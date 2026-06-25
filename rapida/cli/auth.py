import logging
from rapida.az.surgeauth import SurgeTokenCredential
import click



logger = logging.getLogger(__name__)

# cache file path
TOKEN_CACHE_FIlE = 'token_cache.json'


@click.command(short_help="authenticate with UNDP account")
@click.option('-c', '--cache_dir', default=None, type=click.Path(),
              help="Optional. Folder path to store token_cache.json. Default is ~/.rapida folder to store cache file.")

def auth(cache_dir):
    """
    Authenticate with  UNDP account
    """

    credential = SurgeTokenCredential(cache_dir=cache_dir)
    token, exp_ts = credential.get_token("https://storage.azure.com/.default",)
    click.echo(f"Authentication was done successfully")


if __name__ == '__main__':
    auth()