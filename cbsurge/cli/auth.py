import logging
from cbsurge.az.surgeauth import SurgeTokenCredential
import click
from cbsurge.util.setup_logger import setup_logger


logger = logging.getLogger(__name__)

# cache file path
TOKEN_CACHE_FIlE = 'token_cache.json'


@click.command(short_help="authenticate with UNDP account")
@click.option('-c', '--cache_dir', default=None, type=click.Path(),
              help="Optional. Folder path to store token_cache.json. Default is ~/.cbsurge folder to store cache file.")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug"
              )
def auth(cache_dir, debug=False):
    """
    Authenticate with  UNDP account
    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    credential = SurgeTokenCredential(cache_dir=cache_dir)
    token, exp_ts = credential.get_token("https://storage.azure.com/.default",)
    click.echo(f"Authentication was done successfully")


if __name__ == '__main__':
    auth()