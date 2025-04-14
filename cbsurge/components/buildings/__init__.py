from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.util.download_geodata import download_vector
from cbsurge.project.project import  Project
from cbsurge.session import Session
import os
import logging
import pyogrio
from osgeo import gdal, osr
import shapely
from cbsurge.components.buildings.preprocessing import mask_buildings
from cbsurge.util.proj_are_equal import proj_are_equal
import geopandas as gpd


logger = logging.getLogger(__name__)


class BuildingsComponent(Component):

    def __call__(self, variables=None, **kwargs):
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if not var_name in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        project = Project(path=os.getcwd())
        logger.info(f'Assessing component "{self.component_name}" in  {", ".join(project.countries)}')
        with Session() as ses:
            variables_data = ses.get_component(self.component_name)

            for var_name in variables:
                var_data = variables_data[var_name]
                # create instance
                v = BuildingsVariable(
                    name=var_name,
                    component=self.component_name,
                    **var_data
                )


                # assess
                v(**kwargs)

class BuildingsVariable(Variable):

    def __call__(self, *args, **kwargs):
        """
                Assess a variable. Essentially this means a series of steps in a specific order:
                    - download
                    - preprocess
                    - analysis/zonal stats

                :param kwargs:
                :return:
                """

        force_compute = kwargs.get('force_compute', False)
        progress = kwargs.get('progress', None)

        if progress is not None:
            variable_task = progress.add_task(
                description=f'[blue]Assessing {self.component}->{self.name}', total=None)

        if not self.dep_vars:  # simple variable,
            if not force_compute:
                # logger.debug(f'Downloading {self.name} source')
                self.download(**kwargs)
                if progress is not None and variable_task is not None:
                    progress.update(variable_task, description=f'[blue]Downloaded {self.component}->{self.name}')
            else:
                # logger.debug(f'Computing {self.name} using gdal_calc from sources')
                self.compute(**kwargs)
                if progress is not None and variable_task is not None:
                    progress.update(variable_task, description=f'[blue]Computed {self.component}->{self.name}', )

        else:
            if self.operator:
                if not force_compute:
                    # logger.debug(f'Downloading {self.name} from  source')
                    self.download(**kwargs)
                    if progress is not None and variable_task is not None:
                        progress.update(variable_task, description=f'[blue]Downloaded {self.component}->{self.name}')
                else:
                    # logger.info(f'Computing {self.name}={self.sources} using GDAL')
                    self.compute(**kwargs)
                    if progress is not None and variable_task is not None:
                        progress.update(variable_task, description=f'[blue]Computed {self.component}->{self.name}')
            else:
                # logger.debug(f'Computing {self.name}={self.sources} using GeoPandas')
                sources = self.resolve(**kwargs)

        self.evaluate(**kwargs)
        if progress is not None and variable_task is not None:
            progress._tasks[variable_task].total = 1
            progress.update(variable_task, description=f'[blue]Assessed {self.component}->{self.name}', advance=1)

    def download(self, progress=None, force_compute=None, **kwargs):

        logger.debug(f'Downloading {self.name}')
        project = Project(os.getcwd())
        dst_layers = pyogrio.list_layers(project.geopackage_file_path)
        if self.component in dst_layers[:,0]  and not force_compute:
            if 'mask' in dst_layers:
                affected_layer_name = f'{self.component}.affected'
                assert affected_layer_name in dst_layers, f'layer {affected_layer_name}  does not exist'
            self.local_path = f'{project.geopackage_file_path}::{self.component}'
            return
        with gdal.OpenEx(project.geopackage_file_path, gdal.OF_READONLY | gdal.OF_VECTOR) as poly_ds:
            lyr = poly_ds.GetLayerByName(project.polygons_layer_name)
            for ci, country in enumerate(project.countries):
                mode = 'w' if ci == 0 else 'a'
                logger.info(f'Downloading {self.name} in {country}')
                source = self.interpolate_template(template=self.source, country=country, **kwargs)
                source, layer_name = source.split('::')
                try:
                    src_dataset_info = pyogrio.read_info(source, layer_name)
                except Exception as e:
                    logger.error(f'Failed to fetch info on buildings in {country}')
                    if '404' in str(e):
                        logger.error(f'{source} is not reachable')
                    raise
                src_crs = src_dataset_info['crs']
                src_srs = osr.SpatialReference()
                src_srs.SetFromUserInput(src_crs)
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                lyr.SetAttributeFilter(f'iso3="{layer_name}"')
                target_srs = lyr.GetSpatialRef()
                will_reproject = not proj_are_equal(src_srs=src_srs, dst_srs=target_srs)
                if will_reproject:
                    target2src_tr = osr.CoordinateTransformation(target_srs, src_srs)
                mask_polygons = dict()
                for feat in lyr:
                    polygon = feat.GetGeometryRef()
                    if will_reproject:
                        polygon.Transform(target2src_tr)
                    shapely_polygon = shapely.wkb.loads(bytes(polygon.ExportToIsoWkb()))
                    poly_id = feat['h3id'] or feat.GetFID()
                    mask_polygons[poly_id] = shapely_polygon

                download_vector(
                    src_dataset_url=source,
                    src_layer_name=layer_name,
                    dst_dataset_path=project.geopackage_file_path,
                    dst_layer_name=self.component,
                    dst_srs=target_srs,
                    dst_layer_mode=mode,
                    mask_polygons=mask_polygons,
                    progress=progress,
                    add_polyid=True

                )
                lyr.SetAttributeFilter(None)
                lyr.ResetReading()
        self.local_path = f'{project.geopackage_file_path}::{self.component}'
        self._compute_affected_(progress=progress )
        return self.local_path

    def _compute_affected_(self,  progress=None):

        project = Project(os.getcwd())
        if project.raster_mask is not None:

            logger.info('Calculating affected buildings')
            affected_layer_name = f'{self.component}.affected'
            mask_buildings(
                buildings_dataset=project.geopackage_file_path,
                buildings_layer_name=self.component,
                mask_ds_path=project.raster_mask,
                masked_buildings_dataset=project.geopackage_file_path,
                masked_buildings_layer_name=affected_layer_name,
                horizontal_chunks=10,
                vertical_chunks=10,
                workers=4,
                progress=progress,


            )


    def compute(self, force_compute=True, **kwargs):
        assert force_compute, f'invalid force_compute={force_compute}'
        return self.download(force_compute=force_compute, **kwargs)


    def evaluate(self, **kwargs):
        destination_layer = f'stats.{self.component}'
        affected_var_name = f'{self.name}_affected'
        affected_var_percentage_name = f'{affected_var_name}_percentage'
        project = Project(os.getcwd())

        logger.info(f'Evaluating variable {self.name}')

        dataset_path, layer_name = self.local_path.split('::')

        # Read only necessary columns (avoid loading full geometries if not needed)
        buildings_gdf = gpd.read_file(dataset_path, layer=layer_name, columns=['polyid', 'geometry'])

        # List layers once to avoid repeated I/O operations
        layers = pyogrio.list_layers(dataset_path)
        layer_names = layers[:, 0]

        polygons_layer = destination_layer if destination_layer in layer_names else project.polygons_layer_name

        # Read only necessary columns for polygons
        polygons_gdf = gpd.read_file(project.geopackage_file_path, layer=polygons_layer, columns=['h3id', 'geometry'])
        polygons_gdf = polygons_gdf.rename(columns={'h3id': 'polyid'})

        # **Efficient Building Count Calculation**
        if self.name == 'nbuildings':
            var_gdf = buildings_gdf['polyid'].value_counts().reset_index()
            var_gdf.columns = ['polyid', self.name]
        else:
            buildings_gdf[self.name] = buildings_gdf.geometry.area
            var_gdf = buildings_gdf.groupby('polyid', as_index=False)[self.name].sum()

        # **Handle Affected Buildings**
        if project.raster_mask is not None:
            affected_layer_name = f'{self.component}.affected'
            if affected_layer_name in layer_names:
                affected_buildings_gdf = gpd.read_file(dataset_path, layer=affected_layer_name,
                                                       columns=['polyid', 'geometry'])

                if self.name == 'nbuildings':
                    affected_var_gdf = affected_buildings_gdf['polyid'].value_counts().reset_index()
                    affected_var_gdf.columns = ['polyid', affected_var_name]
                else:
                    affected_buildings_gdf[affected_var_name] = affected_buildings_gdf.geometry.area
                    affected_var_gdf = affected_buildings_gdf.groupby('polyid', as_index=False)[affected_var_name].sum()

                # Merge affected data
                var_gdf = var_gdf.merge(affected_var_gdf, on='polyid', how='inner')
                var_gdf[affected_var_percentage_name] = (var_gdf[affected_var_name] / var_gdf[self.name]) * 100

        # **Remove old columns before merging**
        polygons_gdf.drop(columns=[col for col in [self.name, affected_var_name, affected_var_percentage_name] if
                                   col in polygons_gdf.columns], inplace=True)

        # **Final Merge and Save**
        out_gdf = polygons_gdf.merge(var_gdf, on='polyid', how='left')
        out_gdf = out_gdf.rename(columns={'polyid': 'h3id'})

        out_gdf.to_file(dataset_path, layer=destination_layer, driver='GPKG', mode='w')

    def resolve(self, **kwargs):
        pass