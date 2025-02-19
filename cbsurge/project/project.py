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
import rasterio.warp
from osgeo import gdal, ogr, osr
from pyproj import CRS as pCRS
from rasterio.crs import CRS as rCRS
from cbsurge.util.proj_are_equal import proj_are_equal
from cbsurge import constants
from cbsurge.admin.osm import fetch_admin
from cbsurge.az.blobstorage import check_blob_exists
from cbsurge.session import Session
from cbsurge.util.dataset2pmtiles import dataset2pmtiles
from pyogrio import write_dataframe

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

    def __init__(self, path: str,polygons: str = None,
                 mask: str = None, projection: str = 'ESRI:54009',
                 comment: str = None, vector_mask_layer: str = 'mask', **kwargs ):

        if path is None:
            raise ValueError("Project path cannot be None")

        self.path = os.path.abspath(path)
        self.geopackage_file_name = f"{os.path.basename(self.path)}.gpkg"
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
            bbox = None
            target_crs = pCRS.from_user_input(projection)
            target_srs = osr.SpatialReference()
            target_srs.SetFromUserInput(projection)

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

                src_crs = gdf.crs
                src_bbox = tuple(map(float, gdf.total_bounds))
                if not src_crs.is_exact_same(target_crs):
                    gdf.to_crs(crs=target_crs, inplace=True)

                bbox = tuple(map(float,gdf.total_bounds))
                cols = gdf.columns.tolist()
                if not ('h3id' in cols and 'undp_admin_level' in cols):
                    logger.info(f'going to add rapida specific attributes country code')

                    if not src_crs.to_epsg() == 4326:
                        left, bottom, right, top = src_bbox
                        bb = rasterio.warp.transform_bounds(src_crs=rCRS.from_epsg(src_crs.to_epsg()),
                                                              dst_crs=rCRS.from_epsg(4326),
                                                              left=left, bottom=bottom,
                                                              right=right, top=top,

                                                                   )
                    else:
                        bb = src_bbox

                    a0_polygons = fetch_admin(bbox=bb,admin_level=0)
                    a0_gdf = None
                    with io.BytesIO(json.dumps(a0_polygons, indent=2).encode('utf-8') ) as a0l_bio:
                        a0_gdf = geopandas.read_file(a0l_bio).to_crs(crs=target_crs)
                    rgdf_centroids = gdf.copy()
                    rgdf_centroids["geometry"] = gdf.centroid
                    jgdf = geopandas.sjoin(rgdf_centroids, a0_gdf, how="left", predicate="within", )
                    jgdf['geometry'] = gdf['geometry']
                    gdf = jgdf
                self._cfg_['countries'] = tuple(set(gdf['iso3']))

                gdf.to_file(filename=self.geopackage_file_path, driver='GPKG', engine='pyogrio', mode='w', layer='polygons',
                             promote_to_multi=True)


                self.save()

            if mask is not None:
                logger.debug(f'Got mask {mask}')
                raster_mask_local_path = os.path.join(self.data_folder, 'mask.tif')

                try:
                    with gdal.OpenEx(mask, gdal.OF_RASTER|gdal.OF_READONLY) as mds:
                        mds_srs = mds.GetSpatialRef()
                        #mds_epsg = int(mds_srs.GetAuthorityCode(None))
                        creation_options = dict(TILED='YES', COMPRESS='ZSTD', BIGTIFF='IF_SAFER', BLOCKXSIZE=256,
                                                BLOCKYSIZE=256)

                        if not proj_are_equal(mds_srs, target_srs):
                            # reproject raster mask to target projection
                            logger.info(f'Reprojecting  raster mask to {target_crs}')

                            warp_options = gdal.WarpOptions(format='GTiff',
                                                            xRes=constants.DEFAULT_MASK_RESOLUTION_METERS,
                                                            yRes=constants.DEFAULT_MASK_RESOLUTION_METERS,
                                                            dstSRS=projection,creationOptions=creation_options,
                                                            cutlineDSName=self.geopackage_file_path, cutlineLayer='polygons',
                                                            outputBounds=bbox, outputBoundsSRS=projection, outputType=gdal.GDT_Byte,
                                                            srcNodata=None, dstNodata="none", targetAlignedPixels=True)
                            rds = gdal.Warp(destNameOrDestDS=raster_mask_local_path,srcDSOrSrcDSTab=mds, options=warp_options)

                        else:
                            logger.info(f'Ingesting raster mask ')
                            proj_win = bbox[0], bbox[-1], bbox[2], bbox[1]
                            translate_options = gdal.TranslateOptions(
                                format='GTiff',
                                xRes=constants.DEFAULT_MASK_RESOLUTION_METERS,
                                yRes=constants.DEFAULT_MASK_RESOLUTION_METERS,
                                projWin=proj_win,
                                projWinSRS=target_srs,
                                creationOptions=creation_options,
                                noData='None'
                            )

                            rds = gdal.Translate(destName=raster_mask_local_path,srcDS=mds,options=translate_options)

                        with gdal.OpenEx(self.geopackage_file_path, gdal.OF_VECTOR | gdal.OF_UPDATE) as vds:
                            logger.info(f'Polygonizing {raster_mask_local_path} ')
                            mband = rds.GetRasterBand(1)
                            mask_lyr = vds.CreateLayer('mask', geom_type=ogr.wkbMultiPolygon, srs=rds.GetSpatialRef())
                            r = gdal.Polygonize(srcBand=mband, maskBand=mband,outLayer=mask_lyr,iPixValField=-1,
                                                   options = ['-nlt PROMOTE_TO_MULTI', '-makevalid', '-skipinvalid'])
                            assert r == 0, f'Failed to polygonize {raster_mask_local_path}'
                            for feature in mask_lyr:
                                geom = feature.GetGeometryRef()

                                simplified_geom = geom.Simplify(constants.DEFAULT_MASK_POLYGONIZATION_SMOOTHING_BUFFER)  # Use SimplifyPreserveTopology(tolerance) if needed
                                smoothed_geom = simplified_geom.Buffer(constants.DEFAULT_MASK_POLYGONIZATION_SMOOTHING_BUFFER).Buffer(-constants.DEFAULT_MASK_POLYGONIZATION_SMOOTHING_BUFFER)
                                # not cleat if the lines below have real value
                                # if geom.GetGeometryType() == ogr.wkbPolygon:
                                #     print(sm)
                                #     multi_geom = ogr.Geometry(ogr.wkbMultiPolygon)
                                #     multi_geom.AddGeometry(smoothed_geom.Clone())
                                # else:
                                #     multi_geom = smoothed_geom
                                feature.SetGeometry(smoothed_geom)
                                mask_lyr.SetFeature(feature)  # Save changes
                except RuntimeError as e:
                    if mask in str(e):
                        with gdal.OpenEx(mask, gdal.OF_VECTOR|gdal.OF_READONLY) as mds:
                            logger.debug(f'Mask is a vector')
                            lyr = mds.GetLayer(0)
                            lyr_name = lyr.GetName()
                            gdf = geopandas.read_file(mask, layer=lyr_name)
                            target_crs = pCRS.from_user_input(projection)
                            src_crs = gdf.crs
                            if not src_crs.is_exact_same(target_crs):
                                gdf.to_crs(crs=target_crs, inplace=True)
                            pgdf = geopandas.read_file(self.geopackage_file_path, layer='polygons')
                            gdf = geopandas.clip(gdf=gdf, mask=pgdf)
                            gdf.to_file(filename=self.geopackage_file_path, driver='GPKG', engine='pyogrio', mode='w',
                                        layer=vector_mask_layer,
                                        promote_to_multi=True)
                            with io.BytesIO() as bio:
                                vpath = f'/vsimem/{vector_mask_layer}.fgb'
                                write_dataframe(df=gdf, path=bio, layer=vector_mask_layer, driver='FlatGeobuf')
                                gdal.FileFromMemBuffer(vpath, bio.getbuffer())
                                try:
                                    with gdal.OpenEx(vpath) as clipped_mds:
                                        creation_options = dict(TILED='YES', COMPRESS='ZSTD', BIGTIFF='IF_SAFER', BLOCKXSIZE=256, BLOCKYSIZE=256)
                                        rasterize_options = gdal.RasterizeOptions(
                                            format='GTiff', outputType=gdal.GDT_Byte,
                                            creationOptions=creation_options, noData=None, initValues=0,
                                            burnValues=1, layers=[lyr_name],
                                            xRes=constants.DEFAULT_MASK_RESOLUTION_METERS,
                                            yRes=constants.DEFAULT_MASK_RESOLUTION_METERS,
                                            targetAlignedPixels=True,
                                            outputBounds=bbox
                                        )

                                        rds = gdal.Rasterize(destNameOrDestDS=raster_mask_local_path, srcDS=clipped_mds,options=rasterize_options)
                                        rds = None


                                finally:
                                    gdal.Unlink(vpath)

                    else:
                        raise e
    def load_config(self):
        """Load configuration safely to avoid recursion"""
        try:
            with open(self.config_file, mode="r", encoding="utf-8") as f:
                config_data = json.load(f)
            self.__dict__.update(config_data)  # ✅ Update instance variables safely
            self.is_valid
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


