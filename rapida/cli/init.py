import logging
import click
import os
from rapida.session import Session
from rapida.components.population.variables import generate_variables as gen_pop_vars
from rapida.components.buildings.variables import generate_variables as gen_bldgs_vars
from rapida.components.rwi.variables import generate_variables as gen_rwi_vars
from rapida.components.roads.variables import generate_variables as gen_road_vars
from rapida.components.elegrid.variables import generate_variables as gen_electric_vars
from rapida.components.deprivation.variables import generate_variables as gen_depriv_vars
from rapida.components.landuse.variables import generate_variables as gen_landuse_vars
from rapida.components.gdp.variables import generate_variables as gen_gdp_vars
from rapida.util.setup_logger import setup_logger

logger = logging.getLogger(__name__)


AZURE_STORAGE_ACCOUNT=os.environ.get("AZURE_STORAGE_ACCOUNT") or "undpgeohub"
AZURE_PUBLISH_CONTAINER_NAME=os.environ.get("AZURE_PUBLISH_CONTAINER_NAME" ) or "rapida"
AZURE_STAC_CONTAINER_NAME=os.environ.get("AZURE_STAC_CONTAINER_NAME") or "stacdata"
AZURE_FILE_SHARE_NAME=os.environ.get("AZURE_FILE_SHARE_NAME") or  "cbrapida"
GEOHUB_ENDPOINT=os.environ.get("GEOHUB_ENDPOINT") or  "https://geohub.data.undp.org"

def setup_prompt(session: Session):
    ## i have just commented authentication out for now
    # auth = session.authenticate()
    # if auth is None:
    #     if click.confirm("Authentication failed. Do you want to continue initializing the tool? Yes/Enter to continue, No to cancel.", default=True):
    #         click.echo("Initialization will continue without authentication. Please authenticate later.")
    #     else:
    #         click.echo("rapida init was cancelled. Please authenticate later.")
    #         return
    # else:
    #     click.echo("Authentication successful.")
    #
    # click.echo("We need more information to setup from you.")

    # azure blob container setting
    session.set_account_name(AZURE_STORAGE_ACCOUNT)
    logger.debug(f"account name: {session.get_account_name()}")
    session.set_publish_container_name(AZURE_PUBLISH_CONTAINER_NAME)
    logger.debug(f"publish container name: {session.get_publish_container_name()}")
    session.set_stac_container_name(AZURE_STAC_CONTAINER_NAME)
    logger.debug(f"stac container name: {session.get_stac_container_name()}")

    # azure file share setting
    session.set_file_share_name(AZURE_FILE_SHARE_NAME)
    logger.debug(f"file share name: {session.get_file_share_name()}")

    # geohub endpoint setting
    session.set_geohub_endpoint(GEOHUB_ENDPOINT)
    logger.debug(f"GeoHub endpoint: {session.get_geohub_endpoint()}")

    session.save_config()
    vars_dict = {
        "variables": {
            "population":  gen_pop_vars(),
            "buildings": gen_bldgs_vars(),
            "roads": gen_road_vars(),
            "rwi": gen_rwi_vars(),
            "deprivation": gen_depriv_vars(),
            "elegrid": gen_electric_vars(),
            "landuse": gen_landuse_vars(),
            "gdp": gen_gdp_vars(),
        }
    }
    session.config.update(vars_dict)
    session.save_config()
    click.echo(f"Initialization was done. config file was saved to {session.get_config_file_path()}")




@click.command(short_help='initialize RAPIDA tool')
@click.option('--no-input',
              is_flag=True,
              default=False,
              help="Optional. If True, it will automatically answer yes to prompts. Default is False.")
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug"
              )
def init(no_input: bool = False, debug=False):
    """
    Initialize RAPIDA tool. This command will creates a config file under user home directory (~/.cbsurge/config.json) to setup RAPIDA tool.

    Use `--no-input` to disable prompting. Default is False.
    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    click.echo("Welcome to rapida CLI tool!")
    with Session() as session:
        config = session.get_config()
        if not no_input:
            if config:
                if not click.confirm('Your setup has already been done. Would you like to do setup again?', abort=True):
                    return
            else:
                if not click.confirm('Would you like to setup rapida tool?', abort=True):
                    return
    setup_prompt(session)