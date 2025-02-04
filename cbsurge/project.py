import datetime
import os
import click
import logging
import shutil
from osgeo import gdal
from collections import UserDict
import json
import sys


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
    data_folder_name = 'data'
    path = None
    config_file = None
    name = None
    geopackage_file_name = None

    def __init__(self, path:str=None, polygons:str=None,
                 mask:str=None, projection:str='ESRI:54009', comment:str=None, **kwargs):

        self.path = path
        *_, name = self.path.split(os.path.sep)
        self.name = name

        config_vals = dict(
            file_path=self.config_file,
            name=self.name,
            path=self.path,
            config_file=self.config_file,
            create_command=' '.join(sys.argv),
            created_on=datetime.datetime.now().isoformat(),
            user=os.environ.get('USER', os.environ.get('USERNAME')),
        )
        if mask is not None:
            config_vals['mask'] = mask
        if comment is not None:
            config_vals['comment'] = comment
        self.config = Config(**config_vals)



        # path = os.path.abspath(path)
        # config_file = os.path.join(path, self.config_file_name)
        # if os.path.exists(config_file):
        #     self.config = Config(file_path=config_file)
        #     for k, v in self.config.items():
        #         self.__setattr__(k, v)


        # if polygons is not None:
        #     with gdal.OpenEx(polygons) as poly_ds:
        #         lcount = poly_ds.GetLayerCount()
        #         if lcount > 1:
        #             lnames = list()
        #             for i in range(lcount):
        #                 l = poly_ds.GetLayer(i)
        #                 lnames.append(l.GetName())
        #             #click.echo(f'{polygons} contains {lcount} layers: {",".join(lnames)}')
        #             layer_name = click.prompt(
        #                 f'{polygons} contains {lcount} layers: {",".join(lnames)} Please type/select  one or pres enter to skip if you wish to use default value',
        #                 type=str, default=lnames[0])
        #         else:
        #             layer_name = poly_ds.GetLayer(0).GetName()
        #         if not os.path.exists(self.data_folder):
        #             os.makedirs(self.data_folder)
        #         gdal.VectorTranslate(self.geopackage_file, poly_ds, format='GPKG',reproject=True, dstSRS=projection,
        #                          layers=[layer_name], layerName='polygons', geometryType='PROMOTE_TO_MULTI', makeValid=True)
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

    def from_config(self, config_file=None):
        assert  os.path.exists(config_file), f'{config_file} does not exist'
        with open(config_file, "r") as f:
            config_data = json.load(f)

        # self.config = Config(file_path=config_file)
        # for k, v in self.config.items():
        #     self.__setattr__(k, v)
    @property
    def data_folder(self):
        return os.path.join(self.path, self.data_folder_name)

    @property
    def config_file(self):
        return os.path.join(self.path, self.config_file_name)

    @property
    def geopackage_file(self):
        return os.path.join(self.data_folder, self.geopackage_file_name)


    def __str__(self):
        return json.dumps(
            dict(Name=self.name, Path=self.path, Valid=self.is_valid ), indent=4
        )

    @property
    def is_valid(self):
        """Conditions for a valid project"""
        assert os.path.exists(self.path), f'{self.path} does not exist'
        assert os.access(self.path, os.W_OK), f'{self.path} is not writable'
        assert os.path.exists(self.config_file), f'{self.config_file} does not exist'
        assert os.path.getsize(self.config_file) > 0, f'{self.config_file} is empty'
        return True



    def delete(self):
        if click.confirm(f'Are you sure you want to delete {self.name} located in {self.folder} ?', abort=True):
            shutil.rmtree(self.folder)




@click.command(no_args_is_help=True)

@click.argument('name', required=False)
@click.option('-n', '--name', required=True, type=str,
              help='Name representing a new folder in the current directory' )
@click.argument('polygons', required=False)
@click.option('-p', '--polygons', required=True, type=str,
              help='Full path to the vector polygons dataset in any OGR supported format' )
@click.argument('mask', required=False)
@click.option('-m', '--mask', required=False, type=str,
              help='Full path to the mask dataset in any GDAL/OGR supported format. Can be vector or raster' )
@click.argument('comment', required=False)
@click.option('-c', '--comment', required=False, type=str,
              help='Any comment you might want to add into the project config' )

def create(name=None, polygons=None, mask=None, comment=None):
    """
    Create a Rapida project in a new folder

    """
    abs_folder = os.path.abspath(name)
    try:
        shutil.rmtree(abs_folder)
    except FileNotFoundError:
        pass
    if os.path.exists(abs_folder):
        raise FileExistsError(f'Folder "{name}" already exists in {os.getcwd()}')
    else:
        os.mkdir(abs_folder)
    project = Project(path=abs_folder, polygons=polygons, mask=mask, comment=comment)
    assert project.is_valid
    logger.info(f'Project "{project.name}" was created successfully.')




@click.command(no_args_is_help=True)
def info():
    """Info on a Rapida project """
    pass




