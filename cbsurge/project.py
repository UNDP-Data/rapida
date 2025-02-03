import os
import click
import logging
from cbsurge.session import Session, Project
import shutil

logger = logging.getLogger(__name__)


@click.group()
def project():
    f"""Command line interface for {__package__} package"""
    pass


@click.command(no_args_is_help=True)

@click.option('-f', '--folder', required=True, type=str,
              help='Full path to the project folder or a name representing a folder in the current directory' )
@click.option('-p', '--polygons', required=True, type=str,
              help='Full path to the vector polygons dataset in any OGR supported format' )
@click.option('-m', '--mask', required=False, type=str,
              help='Full path to the mask dataset in any GDAL/OGR supported format. Can be vector or raster' )

def create(folder=None, polygons=None, mask=None):
    """
    Create a Rapida project

    """
    with Session() as session:
        afolder = os.path.abspath(folder)
        if os.path.exists(afolder):
            prj = Project(folder=afolder)
            if prj.is_valid:
                logger.info(f'Project {folder} exists at {afolder}. You can continue.')
                prj = Project(folder=folder, polygons=polygons, mask=mask)
                session.config['project'] = prj.folder
                session.save_config()
                return
        if not os.path.isabs(folder):
            if click.confirm(f'Project {folder} will be created in "{os.getcwd()}". Do you want to continue?', abort=True):
                pass

        prj = Project(folder=folder, polygons=polygons, mask=mask)
        session.config['project'] = prj.folder
        session.save_config()



@click.command()
def show():
    """Info on last Rapida project """
    with Session() as session:
        project = session.project
        logger.info(f'Current Rapida project: {project}')

@click.command(no_args_is_help=True)
@click.argument('project', required=True, type=str)
def delete(project):
    """
    Delete an existing Rapida project from file system and main config file

    PROJECT: name of the project

    """

    with Session() as session:
        proj_folder = session.config.get('project', None)

        if proj_folder is not None:
            prj = session.project
            if prj.name == project:
                logger.info(f'Deleting Rapida project {prj.name} located at {proj_folder}')
                shutil.rmtree(proj_folder)
                del session.config['project']
                session.save_config()
        else:
            logger.info(f'The project {project} does not exist in {session.get_config_file_path()}')


project.add_command(show)
project.add_command(create)
project.add_command(delete)

