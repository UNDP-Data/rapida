import asyncio
import logging
import os
from typing import List

from osgeo import gdal
from osgeo_utils.gdal_calc import Calc
from rich.progress import Progress
import geopandas as gpd

from rapida import constants
from rapida.components.landuse.stac import STAC_MAP, interpolate_stac_source, download_stac
from rapida.components.landuse.prediction import predict
from rapida.constants import GTIFF_CREATION_OPTIONS
from rapida.core.component import Component
from rapida.core.variable import Variable
from rapida.project.project import Project
from rapida.session import Session
from rapida.stats.raster_zonal_stats import zst
from rapida.util import geo


logger = logging.getLogger('rapida')


class LanduseComponent(Component):
    def __call__(self, variables: List[str], target_year: int=None, **kwargs):
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
                    target_year=target_year,
                    **var_data
                )
                v(**kwargs)



class LanduseVariable(Variable):

    @property
    def stac_url(self)->str:
        """
        STAC Server root URL
        """
        stac_id = interpolate_stac_source(self.source)['id']
        url = STAC_MAP[stac_id]
        assert url is not None, f'Unsupported stac_id {stac_id}'
        return url

    @property
    def collection_id(self)->str:
        """
        STAC Collection ID
        """
        collection = interpolate_stac_source(self.source)['collection']
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
        value = interpolate_stac_source(self.source)['value']
        return int(value)

    @property
    def target_asset(self) -> dict[str, str]:
        """
        Dictionary of Earth search asset name and band name
        """
        needed_assets = (
            'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12')
        earth_search_assets = (
            'blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'swir16', 'swir22'
        )
        asset_map = dict(zip(earth_search_assets, needed_assets))
        return asset_map

    @property
    def downloaded_files(self) -> List[str]:
        """
        The list of downloaded files for this component
        """
        project = Project(os.getcwd())
        output_dir = os.path.join(os.path.dirname(project.geopackage_file_path), self.component)
        assets = list(self.target_asset.values())
        return [os.path.join(output_dir, f"{asset}.tif") for asset in assets]

    @property
    def prediction_output_image(self) -> str:
        """
        The path to the output image
        """
        project = Project(os.getcwd())
        output_dir = os.path.join(os.path.dirname(project.geopackage_file_path), self.component)
        return os.path.join(output_dir, f"{self.component}_prediction.tif")


    def __init__(self, **kwargs):
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

        self.download(**kwargs)
        self.compute(**kwargs)

        if progress is not None and variable_task is not None:
            progress.update(variable_task, description=f'[blue] Downloaded {self.component}->{self.name}')

        self.evaluate(**kwargs)

        if progress is not None and variable_task is not None:
            progress.update(variable_task, description=f'[blue] Assessed {self.component}->{self.name}')


    def download(self, force=False, **kwargs):
        project = Project(os.getcwd())
        progress: Progress = kwargs.get('progress', None)

        output_dir = os.path.join(os.path.dirname(project.geopackage_file_path), self.component)

        asset_files = self.downloaded_files
        exists = 0
        for asset in asset_files:
            if os.path.exists(asset):
                exists += 1
        if force == False and exists == len(asset_files):
            # if all files already exist, skip download
            pass
        else:
            asyncio.run(download_stac(stac_url=self.stac_url,
                          collection_id=self.collection_id,
                          geopackage_file_path=project.geopackage_file_path,
                          polygons_layer_name=project.polygons_layer_name,
                          output_dir=output_dir,
                          target_year=self.target_year,
                          target_assets=self.target_asset,
                          target_srs=project.target_srs,
                          progress=progress))


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
                            creationOptions=constants.GTIFF_CREATION_OPTIONS,
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
        force = kwargs.get('force', False)
        progress = kwargs.get('progress', None)

        # run the prediction only when the force or prediction image doesn't exist
        if force or not os.path.exists(self.prediction_output_image):
            predict(
                img_paths=self.downloaded_files,
                output_file_path=self.prediction_output_image,
                progress=progress,
            )

        source_value = self.target_band_value

        calc_creation_options = {
            "COMPRESS": "ZSTD",
            "PREDICTOR": 2,
            "BIGTIFF": "IF_SAFER",
            "BLOCKXSIZE": "256",
            "BLOCKYSIZE": "256"
        }

        if progress:
            variable_task = progress.add_task(f"[green]Masking computing land use for variable {self.name}", total=100)

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
            polygons_layer = constants.POLYGONS_LAYER_NAME
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

