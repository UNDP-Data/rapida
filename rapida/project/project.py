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
import country_converter as coco
import requests
import click
import geopandas
import h3.api.basic_int as h3
import pyogrio
from geopandas import GeoDataFrame
from osgeo import gdal, ogr, osr
from pyproj import CRS as pCRS
from azure.storage.fileshare import ShareClient

from rapida import constants
from rapida.admin.osm import fetch_admin
from rapida.az.blobstorage import check_blob_exists, delete_blob
from rapida.session import Session
from rapida.util import geo
from rapida.util.countries import COUNTRY_CODES
from rapida.util.dataset2pmtiles import dataset2pmtiles

logger = logging.getLogger(__name__)
gdal.UseExceptions()


def find_probable_iso3_col(gdf: GeoDataFrame):
    """
    Find the column name that is most likely to contain ISO3 country codes by checking the values.
    If the majority of unique values in a column match known ISO3 codes, return the column name.

    Parameters:
        gdf (GeoDataFrame): Input GeoDataFrame.

    Returns:
        str, integer: The name of the column that most likely contains ISO3 codes, or None if not found.
    """
    df = gdf.convert_dtypes()
    df = df.select_dtypes(include="string")
    potential_cols = dict()
    for column in df.columns:
        # if df[column].dtype == "":
        unique_values = gdf[column].dropna().unique()
        match_count = sum(str(val).upper() in COUNTRY_CODES for val in unique_values)
        if match_count > 0:
            potential_cols[match_count/len(unique_values) * 100] = column
    if len(potential_cols) < 1:
        return None, None
    max_value = max(potential_cols)
    return potential_cols[max_value], max_value


def fetch_ccode(lat=None, lon=None):
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=12&addressdetails=1"
    headers = {"User-Agent": "my-app/1.0 (email@example.com)"}
    response = requests.get(url, headers=headers)
    json_resp = response.json()
    iso2_cc = json_resp.get("address", {}).get("country_code").upper()
    iso3 =coco.convert(names=iso2_cc, to='ISO3')
    return iso3

class Project:
    config_file_name = 'rapida.json'
    data_folder_name = 'data'
    projection: str = 'ESRI:54034'
    _instance = None
    polygons_layer_name = constants.POLYGONS_LAYER_NAME
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path: str,polygons: str = None,
                 mask: str = None,
                 h3id_precision = 7,
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
                    dst_layer=self.polygons_layer_name,
                    target_srs=self.target_srs,
                    access_mode='overwrite'
                )


                gdf = geopandas.read_file(self.geopackage_file_path, layer=self.polygons_layer_name )
                gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
                proj_bounds = tuple(map(float,gdf.total_bounds))

                existing_iso3_column, percentage_match = find_probable_iso3_col(gdf=gdf)

                if existing_iso3_column is not None and existing_iso3_column != 'iso3':
                    gdf = gdf.rename(columns={existing_iso3_column: 'iso3'})

                cols = gdf.columns.tolist()
                if not 'iso3' in cols:
                    logger.info(f'going to add country code into "iso3" column')
                    geo_srs = osr.SpatialReference()
                    geo_srs.ImportFromEPSG(4326)
                    geo_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    coord_trans = osr.CoordinateTransformation(self.target_srs, geo_srs)
                    geo_bounds = coord_trans.TransformBounds(*proj_bounds, 21)

                    admin0_polygons = fetch_admin(bbox=geo_bounds, admin_level=0,
                                                  h3id_precision=h3id_precision)  # using osm admin to populate the iso3 codes

                    target_crs = pCRS.from_user_input(self.projection)
                    with io.BytesIO(json.dumps(admin0_polygons, indent=2).encode('utf-8')) as a0l_bio:
                        a0_gdf = geopandas.read_file(a0l_bio).to_crs(crs=target_crs)
                    centroids = gdf.copy()
                    centroids["geometry"] = gdf.centroid

                    joined = geopandas.sjoin(centroids, a0_gdf, how="left", predicate="within")
                    left_cols = [col for col in centroids.columns if col in joined.columns]
                    left_cols.append('iso3')
                    joined = joined[left_cols]
                    joined['geometry'] = gdf['geometry']

                    invalid_or_missing_mask = joined['iso3'].isna() | ~joined['iso3'].isin(COUNTRY_CODES)

                    if invalid_or_missing_mask.any():
                        logger.info("Missing or invalid ISO3 country codes found. Attempting to assign correct ones...")

                        for idx, row in joined[invalid_or_missing_mask].iterrows():
                            try:
                                centroid = row.geometry.centroid
                                centroid_trans = coord_trans.TransformPoint(centroid.x, centroid.y)
                                lat, lon = centroid_trans[1], centroid_trans[0]
                                new_iso3 = fetch_ccode(lat=lat, lon=lon)
                                if new_iso3:
                                    joined.at[idx, 'iso3'] = new_iso3

                            except Exception as e:
                                logger.warning(f"Failed to fetch ISO3 for geometry at index {idx} ({lat}, {lon}): {e}")

                    self.countries = tuple(sorted(set(filter(lambda x: x in COUNTRY_CODES, joined['iso3']))))
                    cols = joined.columns.tolist()

                    for col_name in ('index_right', 'index_left'):
                        if col_name in cols:
                            joined.drop(columns=[col_name], inplace=True)

                    joined.to_file(filename=self.geopackage_file_path, driver='GPKG', engine='pyogrio', mode='w', layer=self.polygons_layer_name,
                                 promote_to_multi=True, index=False)

                    gdf = joined


                if 'h3id' in cols:
                    h3ids = gdf['h3id'].tolist()
                    # check duplicated
                    if len(h3ids) != len(set(h3ids)):
                        # regenerate h3id
                        gdf['h3id'] = gdf.apply(
                            lambda g: h3.latlng_to_cell(lat=g.geometry.centroid.y, lng=g.geometry.centroid.x,
                                                        res=h3id_precision),
                            axis=1
                        )
                        # save it back
                        gdf.to_file(filename=self.geopackage_file_path, driver='GPKG', engine='pyogrio', mode='w', layer=self.polygons_layer_name,
                                 promote_to_multi=True, index=False)
                else:
                    gdf['h3id'] = gdf.apply(
                        lambda g: h3.latlng_to_cell(lat=g.geometry.centroid.y, lng=g.geometry.centroid.x,
                                                    res=h3id_precision),
                        axis=1
                    )
                    gdf.to_file(filename=self.geopackage_file_path, driver='GPKG', engine='pyogrio', mode='w', layer=self.polygons_layer_name,
                                 promote_to_multi=True, index=False)

                gdf.dropna(subset=['iso3', 'h3id'], inplace=True, axis=0)
                self.countries = tuple(sorted(set(filter(lambda x: x in COUNTRY_CODES, gdf['iso3'].tolist()))))
                self._cfg_['countries'] = self.countries
                self.save()
                gdf.drop(gdf[~gdf['iso3'].isin(COUNTRY_CODES)].index, inplace=True)
                gdf.to_file(filename=self.geopackage_file_path, driver='GPKG', engine='pyogrio', mode='w',
                            layer=self.polygons_layer_name,
                            promote_to_multi=True, index=False)

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
                                      crop_layer_name=self.polygons_layer_name,
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
                        geom_type=ogr.wkbPolygon
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
                        clip_layer=self.polygons_layer_name,
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
    def vector_mask(self):
        mask_layer_name = 'mask'
        layer_list = geopandas.list_layers(self.geopackage_file_path)['name'].tolist()
        if mask_layer_name in layer_list:
            return mask_layer_name
        else:
            return None

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


    def save(self, load_default=True):
        """
        Save rapida.json
        :param load_default: default is True. If True, load current rapida.json file to the memory first.
        """
        os.makedirs(self.data_folder, exist_ok=True)

        if load_default and os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding="utf-8") as cfgf:
                content = cfgf.read()
                data = json.loads(content) if content else {}
        else:
            data = {}

        data.update(self._cfg_)

        with open(self.config_file, 'w', encoding="utf-8") as cfgf:
            json.dump(data, cfgf, indent=4)


    def upload(self, progress=None, overwrite=False, max_concurrency=8):
        """
        Uploads a folder representing a rapida propject to the Azure account and file share set through rapida init

        :param project_folder: str, the full path to the project folder
        :param progress: optional, instance of rich progress to report upload status
        :param overwrite: bool, default = false, whether the files in the project located remotely should be overwritten
        :param max_concurrency: int, the number of threads to use in low level azure api when uploading
        in case they already exists
        :return: None
        """
        with Session() as session:
            account_name = session.get_account_name()
            share_name = session.get_file_share_name()
            account_url = f'https://{account_name}.file.core.windows.net'
            project_name = self._cfg_["name"]
            project_folder = self.path
            with ShareClient(account_url=account_url, share_name=share_name,
                             credential=session.get_credential(), token_intent='backup') as sc:
                with sc.get_directory_client(project_name) as project_dir_client:
                    for root, dirs, files in os.walk(project_folder):
                        directory_name = os.path.relpath(root, project_folder)
                        dc = project_dir_client.get_subdirectory_client(directory_name=directory_name)
                        if not dc.exists():
                            dc.create_directory()
                        for name in files:
                            src_path = os.path.join(root, name)
                            sfc = dc.get_file_client(name)
                            if sfc.exists() and not overwrite:
                                raise FileExistsError(f'{sfc.url} already exists. Set overwrite=True to overwrite')
                            size = os.path.getsize(src_path)
                            with open(src_path, 'rb') as src:
                                if progress:
                                    with progress.wrap_file(src, total=size, description=f'Uploading {name} ', ) as up:
                                        dc.upload_file(name, up, max_concurrency=max_concurrency)
                                        up.progress.remove_task(up.task)
                                else:
                                    dc.upload_file(name, src_path, max_concurrency=max_concurrency)


    def delete(self, no_input=False):
        """
        Delete a project by name from Azure File Share

        :param name: name of the project
        :param no_input: optional, default = false, whether to skip confirmation to answer Yes for all.
        """

        def delete_directory_recursive(sc: ShareClient, dir_name: str):
            """
            Recursively delete all contents inside a directory before deleting the directory itself.
            """
            dir_client = sc.get_directory_client(dir_name)

            for item in dir_client.list_directories_and_files():
                item_path = f"{dir_name}/{item.name}"
                if item.is_directory:
                    delete_directory_recursive(sc, item_path)
                else:
                    file_client = dir_client.get_file_client(item.name)
                    file_client.delete_file()
                    logger.debug(f"Deleted {file_client.url}")
            dir_client.delete_directory()
            logger.debug(f"Deleted {dir_client.url}")


        project_name = self._cfg_["name"]

        if "publish_info" in self._cfg_:
            publish_info = self._cfg_["publish_info"]
            if publish_info:
                dataset_api_url = publish_info.get("geohub_dataset_api")
                if dataset_api_url:
                    try:
                        resp = requests.get(dataset_api_url, timeout=10)

                        if resp.status_code == 200 or resp.status_code == 403:
                            # geohub returns 200 if it is public dataset, and returns 403 if it is private dataset
                            click.echo("This project is likely registered in GeoHub. Please unregister it first before deleting.")
                            return
                        elif resp.status_code == 404:
                            click.echo("This project data was uploaded in Azure, but it is not registered in GeoHub.")
                    except requests.RequestException as e:
                        logger.warning(f"Warning: Failed to check GeoHub registration: {e}")
                        return

                published_files = publish_info.get("publish_files", [])
                if published_files:
                    click.echo("The following files were uploaded for publishing:")
                    for f in published_files:
                        click.echo(f"- {f}")

                    if not no_input and not click.confirm("Do you want to delete these files from Azure Blob Storage?",
                                                          default=False):
                        click.echo("Cancelled deletion of published files.")
                        return
                    else:
                        for blob_url in published_files:
                            blob_deleted = asyncio.run(delete_blob(blob_url))
                            if blob_deleted:
                                logger.debug(f"Deleted {blob_url} successfully")
                        click.echo("Data for publishing was deleted successfully from Azure Blob Container.")
                        del self._cfg_["publish_info"]
                        self.save(load_default=False)

        with Session() as session:
            share_name = session.get_file_share_name()
            account_url = session.get_file_share_account_url()
            with ShareClient(account_url=account_url, share_name=share_name,
                             credential=session.get_credential(), token_intent='backup') as sc:
                # check if the project exists in Azure File Share
                target_project = None
                logger.info(f"Searching for the project '{project_name}' in Azure...")

                for entry in sc.list_directories_and_files(name_starts_with=project_name):
                    if entry.is_directory:
                        if entry.name == project_name:
                            target_project = entry.name

                if target_project is None:
                    logger.warning(f'Project: {project_name} not found in Azure.')
                else:
                    if no_input or click.confirm(f"Project: {project_name} was found. Yes to continue deleting it, or No/Enter to exit. ",
                                     default=False):
                        delete_directory_recursive(sc, target_project)
                        logger.info(f'Successfully deleted the project from Azure: {project_name}.')
                    else:
                        logger.info(f'Cancelled to delete the project from Azure: {project_name}.')

        if no_input or click.confirm(
                f'Do want to continue deleting {self.name} located in {self.path} locally?',
                default=False):
            shutil.rmtree(self.path)
            logger.info(f'Successfully deleted the project folder: {self.path} from local storage.')


    def publish(self, target_layers=None, no_input=False):
        """
        Publish project outcome to Azure blob storage and make data registration URL of GeoHub.

        :param target_layers: optional. If target layers are passed, only they are published. Otherwise, all layers are published.
        :param no_input: optional. If True, it will automatically answer yes to prompts. Default is False.
        """
        project_name = self._cfg_["name"]

        gpkg_path = self.geopackage_file_path
        if not os.path.exists(gpkg_path):
            raise RuntimeError(f"Could not find {gpkg_path} in {self.data_folder}. Please do assess command first.")

        layers = pyogrio.list_layers(gpkg_path)
        layer_names = layers[:, 0]

        if target_layers is None or len(target_layers) == 0:
            target_layers = layer_names

        layer_info = []
        with Session() as session:
            components = session.get_components()
            for c_name in components:
                if c_name in target_layers:
                    variables = session.get_variables(c_name)
                    first_variable_name = next(iter(variables))
                    first_variable = session.get_variable(c_name, first_variable_name)

                    layer_info.append({
                        "layer": f"stats.{c_name}",
                        "license": first_variable["license"],
                        "attribution": first_variable["attribution"],
                    })

            descriptions = [f'**{info["layer"]}** (Data provider: **{info["attribution"]}**, License: **{info["license"]}**)' for info in layer_info]
            joined_descriptions = os.linesep + os.linesep.join(f'- {d}' for d in descriptions)
            description = f"This dataset was generated for the project {self.name} by UNDP RAPIDA tool to assess the following component layers:{joined_descriptions}"
            attributions = [info["attribution"] for info in layer_info]
            attributions.insert(0, "United Nations Development Programme (UNDP)")
            attribution = ", ".join(attributions)

        with Session() as s:
            blob_url = f"az:{s.get_account_name()}:{s.get_publish_container_name()}/projects/{project_name}/{project_name}.pmtiles"

            publish_blob_exists = asyncio.run(check_blob_exists(blob_url))
            if publish_blob_exists:
                if not no_input and not click.confirm(f"Project data was already published at {blob_url}. Do you want to overwrite it? Type yes to proceed, No/Enter to exit. ", default=False):
                    click.echo("Canceled to publish")
                    return

            uploaded_files = None
            try:
                uploaded_files = asyncio.run(dataset2pmtiles(blob_url=blob_url,
                                                             src_file=gpkg_path,
                                                             overwrite=True,
                                                             name=f"RAPIDA: {project_name}",
                                                             description=description,
                                                             attribution=attribution,
                                                             target_layers=target_layers,
                                                             ))
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
                if not no_input:
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