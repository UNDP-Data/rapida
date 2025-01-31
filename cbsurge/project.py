import os
import json
import click
import logging
import shutil
from cbsurge.session import Session

logger = logging.getLogger(__name__)





class Project:
    config_file_name = 'rapida.json'
    data_folder = None
    geopackage_file:str = None
    name:str = None
    def __init__(self, folder:str=None, ):
        folder = os.path.abspath(folder)
        *rest, name = folder.split(os.path.sep)
        self.name = name
        self.folder = folder
        self.data_folder = os.path.join(self.folder, 'data')
        self.geopackage_file = os.path.join(self.folder, self.data_folder, f'{self.name}.gpkg')
        self.config_file = os.path.join(self.folder, self.config_file_name)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            logger.debug(f'Creating project folder ...')
            self.config = dict(
                name=self.name,
                config_file=self.config_file,
                folder=self.folder,
                data_folder=self.data_folder
            )
            logger.debug(f'Creating project config file ...')
            self.serialize()
            logger.info(f'Project {self.name} is located in {self.folder}')
        else:
            if os.path.exists(self.config_file):
                self.config = self.deserialize()
            else:
                logger.info(f'Creating project config file ...')
                self.config = dict(
                    name=self.name,
                    config_file=self.config_file,
                    folder=self.folder,
                    data_folder=self.data_folder
                )
                self.serialize()
                logger.info(f'Project {self.name} is located in {self.folder}')
        assert self.is_valid, f'{self} is not valid'

    def __str__(self):
        txt =   f'''
    name:            {self.name}
    folder:          {self.folder}
    config file:     {self.config_file}
    data folder:     {self.data_folder}
    geopackage file: {self.geopackage_file}                                            
                '''
        return txt

    def deserialize(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as data_h:
                return json.load(data_h)

    def serialize (self):

        with open(self.config_file, "w", encoding="utf-8") as cfg_fh:
            cfg_fh.write(json.dumps(self.config, ensure_ascii=False, indent=4))

    @property
    def is_valid(self):
        assert os.path.exists(self.folder), f'{self.folder} does not exist'
        can_write = os.access(self.folder, os.W_OK)
        proj_cfg_file_exists = os.path.exists(self.config_file)
        proj_cfg_file_is_empty = os.path.getsize(self.config_file) == 0
        geopackage_file_path_is_defined = self.geopackage_file not in (None, '')
        return can_write and proj_cfg_file_exists and not proj_cfg_file_is_empty and geopackage_file_path_is_defined

    def delete(self):
        if click.confirm(f'Are you sure you want to delete {self.name} located in {self.folder} ?', abort=True):
            shutil.rmtree(self.folder)

@click.group()
def project():
    f"""Command line interface for {__package__} package"""
    pass


@click.command(no_args_is_help=True)
@click.argument('folder', required=True, type=click.Path())
def init(folder):
    """
    Create a Rapida project

    FOLDER: the absolute or relative path to the new project

    """
    with Session() as session:
        afolder = os.path.abspath(folder)
        if os.path.exists(afolder):
            prj = Project(folder=afolder)
            if prj.is_valid:
                logger.info(f'Project {folder} exists at {afolder}')
                return
        if not os.path.isabs(folder):
            if click.confirm(f'Project {folder} will be created in "{os.getcwd()}". Do you want to continue?', abort=True):
                pass
        prj = Project(folder=folder)
        session.config['project'] = prj.folder
        session.save_config()


@click.command()
def show():
    """Info on last Rapida project """
    with Session() as session:
        last_project = session.config.get('project', None)
        if last_project is None:
            logger.info(f'No Rapida propject has been defined yet')
        else:
            prj = Project(folder=last_project)
            logger.info(f'Current Rapida project: {prj}')

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
            prj = Project(folder=proj_folder)
            if prj.name == project:
                logger.info(f'Deleting Rapida project {prj.name} located at {proj_folder}')
                shutil.rmtree(proj_folder)
                del session.config['project']
                session.save_config()
        else:
            logger.info(f'The project {project} does not exist in {session.get_config_file_path()}')


project.add_command(show)
project.add_command(init)
project.add_command(delete)

