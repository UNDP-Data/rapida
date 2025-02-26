import asyncio
import datetime
import hashlib
import io
import json
import logging
import os
import shutil
import sys
import webbrowser
import click
import geopandas
from osgeo import gdal, ogr, osr
from pyproj import CRS as pCRS
from cbsurge.util import geo
from cbsurge import constants
from cbsurge.admin.osm import fetch_admin
from cbsurge.az.blobstorage import check_blob_exists
from cbsurge.session import Session
from cbsurge.util.dataset2pmtiles import dataset2pmtiles


logger = logging.getLogger(__name__)
gdal.UseExceptions()


class Project:
    config_file_name = 'rapida.json'
    data_folder_name = 'data'
    projection: str = 'ESRI:54009'
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path: str,polygons: str = None,
                 mask: str = None,
                 comment: str = None, vector_mask_layer: str = 'mask', **kwargs ):

        if path is None:
            raise ValueError("Project path cannot be None")
        self.path = os.path.abspath(path)
        self.geopackage_file_name = f"{os.path.basename(self.path)}.gpkg"
        self.target_srs = osr.SpatialReference()
        self.target_srs.SetFromUserInput(self.projection)
        self.target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not polygons:
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
            # if mask:
            #     self._cfg_['mask'] = mask
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

                geo.import_vector(
                    src_dataset=polygons,
                    src_layer=layer_name,
                    dst_dataset=self.geopackage_file_path,
                    dst_layer=constants.POLYGONS_LAYER_NAME,
                    target_srs=self.target_srs,
                )


                gdf = geopandas.read_file(self.geopackage_file_path, layer=constants.POLYGONS_LAYER_NAME )
                proj_bounds = tuple(map(float,gdf.total_bounds))
                cols = gdf.columns.tolist()
                if not 'iso3' in cols:
                    logger.info(f'going to add country code into "iso3" column')
                    geo_srs = osr.SpatialReference()
                    geo_srs.ImportFromEPSG(4326)
                    geo_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    coord_trans = osr.CoordinateTransformation(self.target_srs, geo_srs)
                    geo_bounds = coord_trans.TransformBounds(*proj_bounds,21)

                    admin0_polygons = fetch_admin(bbox=geo_bounds,admin_level=0)

                    target_crs = pCRS.from_user_input(self.projection)
                    with io.BytesIO(json.dumps(admin0_polygons, indent=2).encode('utf-8') ) as a0l_bio:
                        a0_gdf = geopandas.read_file(a0l_bio).to_crs(crs=target_crs)
                    centroids = gdf.copy()
                    centroids["geometry"] = gdf.centroid
                    joined = geopandas.sjoin(centroids, a0_gdf, how="left", predicate="within")
                    joined['geometry'] = gdf['geometry']

                    self.countries = tuple(set(joined['iso3']))

                    joined.to_file(filename=self.geopackage_file_path, driver='GPKG', engine='pyogrio', mode='w', layer='polygons',
                                 promote_to_multi=True)
                else:
                    self.countries = tuple(set(gdf['iso3']))
                self._cfg_['countries'] = self.countries
                self.save()

            if mask is not None:
                logger.debug(f'Got mask {mask}')

                raster_mask_local_path = os.path.join(self.data_folder, f'{vector_mask_layer}.tif')
                if geo.is_raster(mask):
                    logger.info(f'Got raster mask')
                    '''
                    align raster uses gdalwarp to (1) reproject if there is a need and (2)
                    crop the source with "polygons" layer
                    additionally it can take ANY keyword args that gdalwarp takes.
                    The imported raster mask will have 0 as NODATA in order to avoid
                    operating over nodata pixels when computing affected version of the variable
                    
                    This function could eventually have an a     priori step to make sure the original nodata
                    value is set to zero in case the mask comes with nodata value
                    
                    '''

                    geo.import_raster(target_srs=self.target_srs,
                                      source=mask, dst=raster_mask_local_path,
                                      crop_ds=self.geopackage_file_path,
                                      crop_layer_name=constants.POLYGONS_LAYER_NAME,
                                      return_handle=False,
                                      outputType=gdal.GDT_Byte,
                                      srcNodata=None, dstNodata=0, # this is gdalwarp specific kwrd
                                      targetAlignedPixels=True,

                                      )
                    '''
                    the vector mask will be created now through polygonization
                    it assumes the maks has pixels with value=1 and 0 NODATA. The pixels with value=1
                    are converted to polygons. Additionally are simplified and smoothed.
                    '''
                    geo.polygonize_raster_mask(
                        raster_ds=raster_mask_local_path,
                        dst_dataset=self.geopackage_file_path,
                        dst_layer=vector_mask_layer,
                        geom_type=ogr.wkbMultiPolygon
                    )

                if geo.is_vector(mask):

                    logger.info(f'Got vector mask')
                    # import vector mask
                    geo.import_vector(
                        src_dataset=mask,
                        dst_dataset=self.geopackage_file_path,
                        dst_layer=vector_mask_layer,
                        target_srs=self.target_srs,
                        clip_dataset=self.geopackage_file_path,
                        clip_layer=constants.POLYGONS_LAYER_NAME,
                        access_mode='append'

                    )

                    # rasterize the imported mask
                    geo.rasterize_vector_mask(
                        vector_mask_ds=self.geopackage_file_path,
                        vector_mask_layer=vector_mask_layer,
                        dst_dataset=raster_mask_local_path,
                        nodata_value=0,
                        outputBounds=proj_bounds

                    )


    def load_config(self):
        """Load configuration safely to avoid recursion"""
        try:
            with open(self.config_file, mode="r", encoding="utf-8") as f:
                config_data = json.load(f)
            self.__dict__.update(config_data)  # ✅ Update instance variables safely
            self.is_valid
            # set config data to _cfg_ to ensure other method to access to config dict
            self._cfg_ = config_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config file ({self.config_file}): {e}")

    @property
    def raster_mask(self):
        mask_file = os.path.join(self.data_folder, 'mask.tif' )
        if os.path.exists(mask_file):
            return mask_file
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
        return (os.path.exists(self.path) and os.access(self.path, os.W_OK)
                and os.path.exists(self.config_file)) and os.path.getsize(self.config_file) > 0

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




    def publish(self, yes=False):
        """
        Publish project outcome to Azure blob storage and make data registration URL of GeoHub.

        :param yes: optional. If True, it will automatically answer yes to prompts. Default is False.
        """
        project_name = self._cfg_["name"]

        gpkg_path = self.geopackage_file_path
        if not os.path.exists(gpkg_path):
            raise RuntimeError(f"Could not find {gpkg_path} in {self.data_folder}. Please do assess command first.")

        with Session() as s:
            blob_url = f"az:{s.get_account_name()}:{s.get_publish_container_name()}/projects/{project_name}/{project_name}.pmtiles"

            publish_blob_exists = asyncio.run(check_blob_exists(blob_url))
            if publish_blob_exists:
                if not yes and not click.confirm(f"Project data was already published at {blob_url}. Do you want to overwrite it? Type yes to proceed, No/Enter to exit. ", default=False):
                    click.echo("Canceled to publish")
                    return

            uploaded_files = None
            try:
                uploaded_files = asyncio.run(dataset2pmtiles(blob_url=blob_url, src_file=gpkg_path, overwrite=True))
            except Exception as e:
                logger.error(e)

            if not uploaded_files:
                logger.error(f"Failed to ingest {gpkg_path}")
            else:
                # get PMTiles blob path
                pmtiles_file = next((url for url in uploaded_files if url.endswith(".pmtiles")), None)
                # generate URL for GeoHub publish page
                parts = pmtiles_file.split(":", 2)
                proto, account_name, dst_blob_path = parts
                container_name, *dst_path_parts = dst_blob_path.split(os.path.sep)
                blob_account_url = s.get_blob_service_account_url(account_name=account_name)
                rel_dst_blob_path = os.path.sep.join(dst_path_parts)
                pmtiles_url = f"{blob_account_url}/{container_name}/{rel_dst_blob_path}"

                geohub_endpoint = s.get_geohub_endpoint()
                publish_url = f"{geohub_endpoint}/data/edit?url={pmtiles_url}"

                dataset_id = hashlib.md5(pmtiles_url.encode()).hexdigest()
                dataset_api = f"{geohub_endpoint}/api/datasets/{dataset_id}"

                # save published information to rapida.json
                self._cfg_["publish_info"] = {
                    "publish_pmtiles_url": pmtiles_url,
                    "publish_files": uploaded_files,
                    "geohub_publish_url": publish_url,
                    "geohub_dataset_id": dataset_id,
                    "geohub_dataset_api": dataset_api,
                }
                self.save()
                click.echo(f"Publishing information was stored at {self.config_file}")
                click.echo(f"Open the following URL to register metadata on GeoHub: {publish_url}")

                # if yes, don't open browser.
                if not yes:
                    browser = None
                    try:
                        browser = webbrowser.get()
                    except:
                        click.echo("No browser runnable in this environment. Please copy URL and open the browser manually.")
                    if browser:
                        # if there is a runnable browser, launch a browser to open URL.
                        webbrowser.open(publish_url)


if __name__ == '__main__':

    import logging
    logger = logging.getLogger()

    project_path = '/data/rap/bgd'
    p = Project(path=project_path)
    src_raster = '/data/rap/bgd/data/population/female_active/BGD_female_active_r.tif'
    p.align_raster(source_raster=src_raster)