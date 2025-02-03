import os
import json
import click
import logging
import shutil
from cbsurge.session import Session
from osgeo import gdal, ogr, osr
from collections import UserDict


logger = logging.getLogger(__name__)
gdal.UseExceptions()

class Config(UserDict):
    def __init__(self, file_path, **kwargs):
        """
        Initialize the AutoSaveDict with a target JSON filename.
        If the file exists, load its contents; otherwise, start with an empty dict
        (or with any provided initial values).

        :param file_path: Path to the JSON file where the dictionary will be saved.
        :param args: Positional arguments passed to dict().
        :param kwargs: Keyword arguments passed to dict().
        """
        self.file_path = file_path

        # If the file exists, load its contents
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    loaded_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load JSON from {self.file_path}: {e}")
                loaded_data = {}
        else:
            loaded_data = {}

        # Merge with any additional data provided during instantiation.
        # Values provided in args/kwargs will override those loaded from the file.
        loaded_data.update(dict(**kwargs))

        # Initialize the underlying dictionary with the merged data.
        super().__init__(**loaded_data)
        self._save()  # Write the initial state (in case the file didn't exist)

    def __setitem__(self, key, value):
        """Set the item and save the dictionary to the JSON file."""
        super().__setitem__(key, value)
        self._save()

    def __delitem__(self, key):
        """Delete the item and save the dictionary to the JSON file."""
        super().__delitem__(key)
        self._save()

    def update(self, *args, **kwargs):
        """Update the dictionary and save the changes."""
        super().update( *args, **kwargs)
        self._save()

    def clear(self):
        """Clear the dictionary and save the changes."""
        super().clear()
        self._save()

    def pop(self, key, *args):
        """Remove the specified key and save the dictionary."""
        value = super().pop(key, *args)
        self._save()
        return value

    def popitem(self):
        """Remove and return an arbitrary (key, value) pair and save the dictionary."""
        item = super().popitem()
        self._save()
        return item

    def _save(self):
        """Write the current state of the dictionary to the JSON file."""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            print(f"Error saving AutoSaveDict to {self.file_path}: {e}")


class Project:
    config_file_name = 'rapida.json'
    data_folder = None
    geopackage_file:str = None
    name:str = None
    def __init__(self, folder:str=None, polygons:str=None, mask:str=None, projection='ESRI:54009' ):

        folder = os.path.abspath(folder)
        config_file = os.path.join(folder, self.config_file_name)
        if os.path.exists(config_file):
            self.config = Config(file_path=config_file)
            for k, v in self.config.items():
                self.__setattr__(k, v)

        else:
            self.folder = folder
            *rest, name = self.folder.split(os.path.sep)
            self.name = name
            self.config_file = os.path.join(self.folder, self.config_file_name)
            self.data_folder = os.path.join(self.folder, 'data')
            self.geopackage_file = os.path.join(self.data_folder, f'{self.name}.gpkg')
            if not os.path.exists(self.folder):
                logger.debug(f'Creating project folder ...')
                os.makedirs(self.folder)
                self.config = Config(file_path=self.config_file,
                                     name=self.name,
                                     config_file=self.config_file,
                                     folder=self.folder,

                                     )
                logger.debug(f'Creating project config file ...')
                logger.info(f'Project {self.name} was created in {self.folder}')
            else:
                if os.path.exists(self.config_file):
                    self.config = Config(file_path=self.config_file)

                else:
                    logger.info(f'Creating project config file ...')
                    self.config = Config(file_path=self.config_file,
                                         name=self.name,
                                         config_file=self.config_file,
                                         folder=self.folder,
                                         )

                    logger.info(f'Project {self.name} is located in {self.folder}')

        if polygons is not None:
            with gdal.OpenEx(polygons) as poly_ds:
                lcount = poly_ds.GetLayerCount()
                if lcount > 1:
                    lnames = list()
                    for i in range(lcount):
                        l = poly_ds.GetLayer(i)
                        lnames.append(l.GetName())
                    #click.echo(f'{polygons} contains {lcount} layers: {",".join(lnames)}')
                    layer_name = click.prompt(
                        f'{polygons} contains {lcount} layers: {",".join(lnames)} Please type/select  one or pres enter to skip if you wish to use default value',
                        type=str, default=lnames[0])
                else:
                    layer_name = poly_ds.GetLayer(0).GetName()
                if not os.path.exists(self.data_folder):
                    os.makedirs(self.data_folder)
                gdal.VectorTranslate(self.geopackage_file, poly_ds, format='GPKG',reproject=True, dstSRS=projection,
                                 layers=[layer_name], layerName='polygons', geometryType='PROMOTE_TO_MULTI', makeValid=True)
        # if mask is not  None:
        #     try:
        #         vm_ds = gdal.OpenEx(mask, gdal.OF_VECTOR)
        #     except RuntimeError as ioe:
        #         if 'supported' in str(ioe):
        #             vm_ds = None
        #         else:
        #             raise
        #     if vm_ds is not None:
        #         pass
        #
        # assert self.is_valid, f'{self} is not valid'




    def __str__(self):
        txt =   f'''
    name:            {self.name}
    folder:          {self.folder}
    config file:     {self.config_file}
    data folder:     {self.data_folder}
    geopackage file: {self.geopackage_file}
                '''
        return txt

    @property
    def is_valid(self):
        assert os.path.exists(self.folder), f'{self.folder} does not exist'
        can_write = os.access(self.folder, os.W_OK)
        proj_cfg_file_exists = os.path.exists(self.config_file)
        proj_cfg_file_is_empty = os.path.getsize(self.config_file) == 0
        geopackage_file_path_is_defined = self.geopackage_file not in (None, '')
        return can_write and proj_cfg_file_exists and not proj_cfg_file_is_empty

    def delete(self):
        if click.confirm(f'Are you sure you want to delete {self.name} located in {self.folder} ?', abort=True):
            shutil.rmtree(self.folder)

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
        last_project = session.config.get('project', None)
        if last_project is None:
            logger.info(f'No Rapida project has been defined yet')
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
project.add_command(create)
project.add_command(delete)

