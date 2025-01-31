import os
import json
import click
import logging

logger = logging.getLogger(__name__)





class Project:
    config_file_name = 'rapida.json'
    data_folder = 'data'
    name:str = None
    def __init__(self, folder:str=None, ):
        fldr = os.path.abspath(folder)
        *rest, name = fldr.split(os.path.sep)
        self.name = name
        self.folder = fldr
        self.config_file = os.path.join(self.folder, self.config_file_name)
        if not os.path.exists(self.folder):
            logger.info(f'Going to create {self.folder}')
            os.makedirs(fldr)
            logger.info(f'Going to create {self.config_file}')
            self.config = dict(
                name=self.name,
                folder=self.folder,
                data_folder=self.data_folder
            )
            self.serialize()
        else:
            if os.path.exists(self.config_file):
                self.config = self.deserialize()
            else:
                logger.info(f'Going to create {self.config_file}')
                self.config = dict(
                    name=self.name,
                    folder=self.folder,
                    data_folder=self.data_folder
                )
                self.serialize()
        assert self.is_valid, f'{self} is not valid'
        logger.info(f'Current Rapida project {self.name} is set to {self.folder}')
    def __str__(self):
        return f'Rapida project {self.name}'

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
        return can_write and proj_cfg_file_exists and not proj_cfg_file_is_empty



@click.command()
@click.argument('folder', required=False, type=click.Path())

def project(folder):
    """Create/open a project """
    project_folder = folder or os.getcwd()
    project = Project(folder=project_folder)

