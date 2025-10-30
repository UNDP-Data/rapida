import asyncio
import datetime
import json
import logging
import os
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Event
from concurrent.futures import  as_completed
from typing import List

import geopandas
import rasterio

from osgeo import gdal
from osgeo_utils.gdal_calc import Calc
from rich.progress import Progress, TimeElapsedColumn
import geopandas as gpd
from rapida.components.landuse.search_utils.s2item import Sentinel2Item
from rapida.components.landuse.constants import SENTINEL2_BAND_MAP
from rapida.components.landuse.download import download_stac, find_sentinel_imagery
from rapida.components.landuse.constants import STAC_MAP
from rapida.components.landuse.sentinel_item import SentinelItem
from rapida.constants import GTIFF_CREATION_OPTIONS, POLYGONS_LAYER_NAME
from rapida.core.component import Component
from rapida.core.variable import Variable
from rapida.project.project import Project
from rapida.session import Session
from rapida.stats.raster_zonal_stats import zst
from rapida.util import geo

from rapida.components.landuse.search_utils.search import fetch_s2_tiles



logger = logging.getLogger('rapida')


class LanduseComponent(Component):

    def __call__(self, variables: List[str], datetime_range: str=None, cloud_cover:int = None, **kwargs):
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if var_name not in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        with Session() as session:
            variable_data = session.get_component(self.component_name)

            for var_name in variables:
                var_data = variable_data[var_name]

                v = LanduseVariable(
                    name=var_name,
                    component=self.component_name,
                    datetime_range=datetime_range,
                    cloud_cover=cloud_cover,
                    **var_data
                )
                v(**kwargs)



class LanduseVariable(Variable):

    @property
    def stac_url(self)->str:
        """
        STAC Server root URL
        """
        stac_id = self._interpolate_stac_source(self.source)['id']
        url = STAC_MAP[stac_id]
        assert url is not None, f'Unsupported stac_id {stac_id}'
        return url

    @property
    def collection_id(self)->str:
        """
        STAC Collection ID
        """
        collection = self._interpolate_stac_source(self.source)['collection']
        return collection

    @property
    def affected_path(self):
        path, file_name = os.path.split(self.local_path)
        fname, ext = os.path.splitext(file_name)
        return os.path.join(path, f'{fname}_affected{ext}')


    @property
    def target_band_value(self)->int:
        """
        Target band value for zonal statistics
        """
        value = self._interpolate_stac_source(self.source)['value']
        return int(value)


    @property
    def prediction_output_image(self) -> str:
        """
        The path to the output image
        """
        project = Project(os.getcwd())
        output_dir = os.path.join(os.path.dirname(project.geopackage_file_path), self.component)
        start_dates, end_dates = self.datetime_range.split('/')
        start_date = datetime.datetime.fromisoformat(start_dates)
        end_date = datetime.datetime.fromisoformat(end_dates)
        return os.path.join(output_dir, f"{self.component}_prediction_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.tif")


    def __init__(self, **kwargs):
        kwargs.update({'s2_tiles':{}}) #  only way to patch because of pydantic
        super().__init__(**kwargs)
        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        output_filename = f"{self.name}.tif"
        self.local_path = os.path.join(os.path.dirname(geopackage_path), self.component, output_filename)


    def __call__(self, *args, **kwargs):
        progress: Progress = kwargs.get('progress', None)
        variable_task = None
        if progress is not None:
            variable_task = progress.add_task(
                description=f'[blue] Assessing {self.component}->{self.name}', total=None)

        try:
            self.download(**kwargs)

            #self.compute(**kwargs)

            if progress is not None and variable_task is not None:
                progress.update(variable_task, description=f'[blue] Downloaded {self.component}->{self.name}')

            #self.evaluate(**kwargs)

            if progress is not None and variable_task is not None:
                progress.update(variable_task, description=f'[blue] Assessed {self.component}->{self.name}')
        except Exception as e:
            raise e
        finally:
            if variable_task is not None:
                progress.remove_task(variable_task)


    def download(self, force=False, **kwargs):

        project = Project(os.getcwd())


        start_date, end_date = self.datetime_range.split('/')
        progress: Progress = kwargs.get('progress', None)
        stop = Event()
        total_n_files = 0
        s2_images = {}

        if force or not os.path.exists(self.prediction_output_image):

            s2_tiles_dict = fetch_s2_tiles(stac_url=self.stac_url, bbox=project.geobounds,
                                     start_date=start_date, end_date=end_date,
                                     max_cloud_cover=self.cloud_cover, progress=progress, prune=True,#filter_for_dev=['36MWE']
                                     )


            for k, v in s2_tiles_dict.items():
                total_n_files+= len(v) * len(SENTINEL2_BAND_MAP)

            output_dir = os.path.dirname(self.prediction_output_image)
            os.makedirs(output_dir, exist_ok=True)
            download_task = None
            if progress:
                download_task = progress.add_task(f"[cyan]Downloading {total_n_files} Sentinel2 images in {len(s2_tiles_dict)} grids ", total=len(s2_tiles_dict))

            failed= {}
            ndone = 0
            downloaded = {}
            with ThreadPoolExecutor(max_workers=5, ) as tpe:
                jobs = dict()
                for mgrs_grid_id, candidates in s2_tiles_dict.items():
                    #print(json.dumps(candidates[0].assets, indent=4))
                    s2i = Sentinel2Item(mgrs_grid=mgrs_grid_id, s2_tiles=candidates, workdir=output_dir, target_crs=project.projection)
                    self.s2_tiles[mgrs_grid_id] = s2i
                    jobs[tpe.submit(s2i.download, bands=s2i.bands, progress=progress, force=force, )] = mgrs_grid_id

                try:
                    for future in as_completed(jobs):
                        grid = jobs[future]
                        try:
                            downloaded_files = future.result()
                            ndone+=1
                            if progress is not None and download_task is not None:
                                progress.update(download_task,advance=1)
                        except Exception as e:
                            failed[grid] = e
                            if progress is not None and download_task is not None:
                                progress.update(download_task, advance=1)
                            raise
                        downloaded[grid] = downloaded_files
                except KeyboardInterrupt:
                    stop.set()
                    for s2i in self.s2_tiles.values():
                        loop = getattr(s2i, "_loop", None)
                        task = getattr(s2i, "_task", None)
                        if loop and task and not task.done():
                            try:
                                loop.call_soon_threadsafe(task.cancel)
                            except RuntimeError:
                                pass
                    # this only cancels pending (not running) futures, still good to call:
                    tpe.shutdown(wait=False, cancel_futures=True)
                    raise

                finally:

                    if progress is not None and download_task is not None:
                        progress.update(download_task,
                                        #description=f'[red]Downloaded Sentinel2 imagery in {ndone} MGRS grids',
                                        advance=1)

            for grid, err in failed.items():
                logger.error(f'Failed to download S2 imagery in {grid} because {err}')
                # it is debatable if the error is to be swallowed or propagated


            for mgrs_grid, s2itm in self.s2_tiles.items():
                for band, vrt in s2itm.vrts.items():
                    if vrt is None:
                        continue
                    if band not in s2_images:s2_images[band] = []
                    s2_images[band].append(vrt)

            vrts = []
            for band, vrt_files in s2_images.items():
                #opts = gdal.BuildVRTOptions()
                band_vrt = os.path.join(output_dir, f'{band}.vrt')

                #band_vrt = f'/vsimem/{band}.vrt'

                with gdal.BuildVRT(destName=band_vrt, srcDSOrSrcDSTab=vrt_files) as bvrt:
                    bvrt.FlushCache()
                    vrts.append(band_vrt)





    def download_old(self, force=False, **kwargs):
        project = Project(os.getcwd())
        progress: Progress = kwargs.get('progress', None)

        if force or not os.path.exists(self.prediction_output_image):
            loop = asyncio.get_event_loop()
            try:
                run = loop.create_task(download_stac(stac_url=self.stac_url,
                              collection_id=self.collection_id,
                              geopackage_file_path=project.geopackage_file_path,
                              polygons_layer_name=project.polygons_layer_name,
                              output_file=self.prediction_output_image,
                              target_srs=project.target_srs,
                              datetime_range=self.datetime_range,
                              cloud_cover=self.cloud_cover,

                              progress=progress))
                loop.run_until_complete(run)
            except (KeyboardInterrupt, asyncio.CancelledError) as e:
                logger.info(f'Cancelled download task for {self.name}')
                run.cancel()
                raise


    def _compute_affected_(self, **kwargs):
        if geo.is_raster(self.local_path):
            project = Project(os.getcwd())
            if project.raster_mask:

                affected_local_path = self.affected_path
                # create a temporary mask with resolution 10 by 10

                # get resolution from local_path
                with gdal.Open(self.local_path, gdal.GA_ReadOnly) as ds:

                    if ds is not None:
                        geotransform = ds.GetGeoTransform()
                        x_res = geotransform[1]
                        y_res = abs(geotransform[5])


                    vrt_path = '/vsimem/warped_project_mask.vrt'

                    options = gdal.BuildVRTOptions(
                        outputBounds=(geotransform[0], geotransform[3] + ds.RasterYSize * geotransform[5],
                                      geotransform[0] + ds.RasterXSize * geotransform[1], geotransform[3]),
                        xRes=x_res,
                        yRes=y_res,

                    )
                    with gdal.BuildVRT(vrt_path, [project.raster_mask], options=options) as vrt:
                        warp_options = dict(
                            format='GTiff',
                            xRes=x_res,
                            yRes=y_res,
                            creationOptions=GTIFF_CREATION_OPTIONS,
                            outputBounds=(geotransform[0], geotransform[3] + ds.RasterYSize * geotransform[5],
                                          geotransform[0] + ds.RasterXSize * geotransform[1], geotransform[3]),
                        )


                        warped_project_mask = project.raster_mask.replace('mask.tif', 'warped_mask.tif')
                        temp_mask_ds = gdal.Warp(
                            destNameOrDestDS=warped_project_mask,
                            srcDSOrSrcDSTab=project.raster_mask,
                            options=gdal.WarpOptions(**warp_options),
                            outputType=gdal.GDT_Byte,
                            outputSRS=project.target_srs,
                            targetAlignedPixels=True,
                            multiThread=True,
                        )
                        temp_mask_ds = None


                    calc_ds = Calc(calc='A*B',
                              outfile=affected_local_path,
                              projectionCheck=True,
                              format='GTiff',
                              creation_options=GTIFF_CREATION_OPTIONS,
                              quiet=False,
                              A=self.local_path,
                              B=warped_project_mask,
                              overwrite=True,
                              NoDataValue=None,
                            )
                    calc_ds = None
                assert os.path.exists(self.affected_path), f'Failed to compute {self.affected_path}'

                return affected_local_path

    def compute(self, **kwargs):
        progress = kwargs.get('progress', None)
        variable_task = None
        if progress:
            variable_task = progress.add_task(f"[red]Creating variable {self.name}", total=100)

        source_value = self.target_band_value


    def compute_old(self, **kwargs):
        progress = kwargs.get('progress', None)
        variable_task = None
        if progress:
            variable_task = progress.add_task(f"[green]Masking computing land use for variable {self.name}", total=100)

        source_value = self.target_band_value

        calc_creation_options = {
            "COMPRESS": "ZSTD",
            "PREDICTOR": 2,
            "BIGTIFF": "IF_SAFER",
            "BLOCKXSIZE": "256",
            "BLOCKYSIZE": "256"
        }

        def progress_callback(complete, message, user_data):
            if progress and variable_task is not None:
                progress.update(variable_task, completed=int(complete * 100))
            return 1

        # create a gdal expression for where(A == source_value, 10, 0)
        # I set the value to 100 because the resolution is 10 by 10 so area of each pixel would be 100
        expression = f"where(A == {source_value}, 100, 0)"

        ds = Calc(
            calc=expression,
            outfile=self.local_path,
            projectionCheck=True,
            format='GTiff',
            creation_options=calc_creation_options,
            overwrite=True,
            A=self.prediction_output_image,
            NoDataValue=0,
            progress_callback=progress_callback
        )

        ds = None

        # Then, compute affected area for land use
        self._compute_affected_(**kwargs)

        if variable_task is not None:
            progress.remove_task(variable_task)

    def evaluate(self, **kwargs):
        progress: Progress = kwargs.get('progress', Progress())
        evaluate_task = None
        if progress is not None:
            evaluate_task = progress.add_task(
                description=f'[red] Going to evaluate {self.name} in {self.component} component', total=None)
        dst_layer = f'stats.{self.component}'

        project = Project(path=os.getcwd())
        layers = gpd.list_layers(project.geopackage_file_path)
        lnames = layers.name.tolist()
        if dst_layer in lnames:
            polygons_layer = dst_layer
        else:
            polygons_layer = POLYGONS_LAYER_NAME
        if self.operator:
            assert os.path.exists(self.local_path), f'{self.local_path} does not exist'

            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task, description=f'[red] Evaluating variable {self.name} using zonal stats')

            # raster variable, run zonal stats
            src_rasters = [self.local_path]
            var_ops = [(self.name, self.operator)]

            if project.raster_mask is not None:
                affected_local_path = self.affected_path
                src_rasters.append(affected_local_path)
                var_ops.append((f'{self.name}_affected', self.operator))

            gdf = zst(src_rasters=src_rasters,
                      polygon_ds=project.geopackage_file_path,
                      polygon_layer=polygons_layer, vars_ops=var_ops
                      )

            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task, description=f'[red] Evaluated variable {self.name} using zonal stats')

            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task,
                                description=f'[red] Writing {self.name} to {project.geopackage_file_path}:{dst_layer}')

            if project.raster_mask and os.path.exists(self.affected_path) and self.percentage:
                affected_var_name = f'{self.name}_affected'
                affected_var_percentage_name = f'{affected_var_name}_percentage'
                gdf[affected_var_name] = gdf[affected_var_name].fillna(0)
                gdf[affected_var_percentage_name] = gdf[affected_var_name] / gdf[self.name] * 100
                gdf[affected_var_percentage_name] = gdf[affected_var_percentage_name].fillna(0)

            gdf.to_file(project.geopackage_file_path, layer=dst_layer, driver="GPKG", overwrite=True)
        else:
            progress.update(evaluate_task,
                            description=f'[red] {self.name} was skipped because of lack of operator definition.')

        if progress and evaluate_task:
            progress.remove_task(evaluate_task)
        pass

    def resolve(self, **kwargs):
        pass

    def _interpolate_stac_source(self, source: str) -> dict[str, str]:
        """
        Interpolate stac source. Source of stac should be defined like below:

        {stac_id}:{collection_id}:{target band value}

        :param source: stac source
        :return: dist consist of id, collection and value
        """
        parts = source.split(':')
        assert len(parts) == 3, 'Invalid source definition'
        stac_id, collection, target_value = parts
        return {
            'id': stac_id,
            'collection': collection,
            'value': target_value
        }
