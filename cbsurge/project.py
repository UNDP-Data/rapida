import os
import json
import click
import logging

logger = logging.getLogger(__name__)





class Project:
    config_file_name = 'rapida.json'
    data_folder = None
    geopackage_file:str = None
    name:str = None
    def __init__(self, folder:str=None, ):
        fldr = os.path.abspath(folder)
        *rest, name = fldr.split(os.path.sep)
        self.name = name
        self.folder = fldr
        self.data_folder = os.path.join(self.folder, 'data')
        self.geopackage_file = os.path.join(self.folder, self.data_folder, f'{self.name}.gpkg')
        self.config_file = os.path.join(self.folder, self.config_file_name)
        if not os.path.exists(self.folder):
            os.makedirs(fldr)
            logger.info(f'Creating project folder ...')
            self.config = dict(
                name=self.name,
                config_file=self.config_file,
                folder=self.folder,
                data_folder=self.data_folder
            )
            logger.info(f'Creating project config file ...')
            self.serialize()

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
        assert self.is_valid, f'{self} is not valid'
        logger.info(self)

    def __str__(self):
        txt =   f'''Current Rapida project: 
    name:            {self.name}
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

    def wipe(self):
        click.confirm(f'')

@click.command()
@click.argument('folder', required=False, type=click.Path())

def project(folder):
    """Create/open a project """
    project_folder = folder or os.getcwd()
    project = Project(folder=project_folder)


