import datetime
import os
import click
import logging
import shutil
from osgeo import gdal
import geopandas
import json
import sys
from pyproj import CRS
from cbsurge.isobbox import ll2iso3


logger = logging.getLogger(__name__)
gdal.UseExceptions()


class Project:
    config_file_name = 'rapida.json'
    data_folder_name = 'data'
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path: str, polygons: str = None,
                 mask: str = None, projection: str = 'ESRI:54009',
                 comment: str = None, save=True, **kwargs):

        if path is None:
            raise ValueError("Project path cannot be None")

        self.path = os.path.abspath(path)
        self.geopackage_file_name = f"{os.path.basename(self.path)}.gpkg"

        if os.path.exists(self.config_file):
            self.load_config()  # ✅ Call a function that loads config safely
        else:
            self.name = os.path.basename(self.path)
            self._cfg_ = {
                "name": self.name,
                "path": self.path,
                "config_file": self.config_file,
                "create_command": ' '.join(sys.argv),
                "created_on": datetime.datetime.now().isoformat(),
                "user": os.environ.get('USER', os.environ.get('USERNAME')),
            }
            if mask:
                self._cfg_['mask'] = mask
            if comment:
                self._cfg_['comment'] = comment

            if polygons is not None:
                l = geopandas.list_layers(polygons)
                lnames = l.name.tolist()
                lcount = len(lnames)
                if lcount > 1:
                    click.echo(f'{polygons} contains {lcount} layers: {",".join(lnames)}')
                    layer_name = click.prompt(
                        f'{polygons} contains {lcount} layers: {",".join(lnames)} Please type/select  one or pres enter to skip if you wish to use default value',
                        type=str, default=lnames[0])
                else:
                    layer_name = lnames[0]
                if not os.path.exists(self.data_folder):
                    os.makedirs(self.data_folder)
                gdf = geopandas.read_file(polygons, layer=layer_name, )
                rgdf = gdf.to_crs(crs=CRS.from_user_input(projection))
                cols = rgdf.columns.tolist()
                if not ('h3id' in cols and 'undp_admin_level' in cols):
                    logger.info(f'going to add ISO3 country code')
                    c = gdf.to_crs(epsg=4326).centroid
                    rgdf["iso3"] = c.apply(lambda point: ll2iso3(point.y, point.x))
                    self._cfg_['countries'] = list(set(rgdf['iso3']))

                rgdf.to_file(filename=self.geopackage_file_path, driver='GPKG', engine='pyogrio', mode='w', layer='polygons',
                             promote_to_multi=True)


            if save:
                self.save()

    def load_config(self):
        """Load configuration safely to avoid recursion"""
        try:
            with open(self.config_file, mode="r", encoding="utf-8") as f:
                config_data = json.load(f)
            self.__dict__.update(config_data)  # ✅ Update instance variables safely
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config file ({self.config_file}): {e}")

    @property
    def data_folder(self):
        return os.path.join(self.path, self.data_folder_name)

    @property
    def config_file(self):
        return os.path.join(self.path, self.config_file_name)

    @property
    def geopackage_file_path(self):
        return os.path.join(self.data_folder, self.geopackage_file_name)

    def __str__(self):
        return json.dumps(
            {"Name": self.name, "Path": self.path, "Valid": self.is_valid}, indent=4
        )

    @property
    def is_valid(self):
        """Conditions for a valid project"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"{self.path} does not exist")
        if not os.access(self.path, os.W_OK):
            raise PermissionError(f"{self.path} is not writable")
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"{self.config_file} does not exist")
        if os.path.getsize(self.config_file) == 0:
            raise ValueError(f"{self.config_file} is empty")
        return True

    def delete(self, force=False):
        if not force and not click.confirm(f'Are you sure you want to delete {self.name} located in {self.path}?',
                                           abort=True):
            return

        shutil.rmtree(self.path)

    def save(self):
        os.makedirs(self.data_folder, exist_ok=True)

        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding="utf-8") as cfgf:
                content = cfgf.read()
                data = json.loads(content) if content else {}
        else:
            data = {}

        data.update(self._cfg_)

        with open(self.config_file, 'w', encoding="utf-8") as cfgf:
            json.dump(data, cfgf, indent=4)


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
        logger.error(f'Folder "{name}" already exists in {os.getcwd()}')
        sys.exit(1)
    else:
        os.mkdir(abs_folder)
    project = Project(path=abs_folder, polygons=polygons, mask=mask, comment=comment)
    assert project.is_valid
    logger.info(f'Project "{project.name}" was created successfully.')




@click.command(no_args_is_help=True)
def info():
    """Info on a Rapida project """
    pass




