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

class Project:


    config_file_name = 'rapida.json'
    data_folder_name = 'data'
    path = None
    config_file = None
    name = None
    geopackage_file_name = None

    def __init__(self, path:str=None, polygons:str=None,
                 mask:str=None, projection:str='ESRI:54009', comment:str=None, save=True,  **kwargs):
        self.path = path

        *_, name = self.path.split(os.path.sep)
        self.name = name

        self._cfg_ = dict(
            file_path=self.config_file,
            name=self.name,
            path=self.path,
            config_file=self.config_file,
            create_command=' '.join(sys.argv),
            created_on=datetime.datetime.now().isoformat(),
            user=os.environ.get('USER', os.environ.get('USERNAME')),
        )
        if mask is not None:
            self._cfg_['mask'] = mask
        if comment is not None:
            self._cfg_['comment'] = comment


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
        if save: self.save()
    @classmethod
    def from_config(cls, config_file=None):
        assert  os.path.exists(config_file), f'{config_file} does not exist'
        with open(config_file, mode="r", encoding="utf-8") as f:
            config_data = json.load(f)
            return cls(save=False,**config_data)

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
        if click.confirm(f'Are you sure you want to delete {self.name} located in {self.path} ?', abort=True):
            shutil.rmtree(self.path)


    def save(self):
        with open(self.config_file, mode='a+', encoding="utf-8") as cfgf:
            content = cfgf.read()
            data = json.loads(content) if content else {}
            data.update(self._cfg_)
            json.dump(data, fp=cfgf, indent=4)


@click.command(no_args_is_help=True)
@click.option('-n', '--name', required=True, type=str,
              help='Name representing a new folder in the current directory' )
@click.option('-p', '--polygons', required=True, type=str,
              help='Full path to the vector polygons dataset in any OGR supported format' )
@click.option('-m', '--mask', required=False, type=str,
              help='Full path to the mask dataset in any GDAL/OGR supported format. Can be vector or raster' )
@click.option('-c', '--comment', required=False, type=str,
              help='Any comment you might want to add into the project config' )

def create(name=None, polygons=None, mask=None, comment=None):
    """
    Create a Rapida project in a new folder

    """
    abs_folder = os.path.abspath(name)
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




