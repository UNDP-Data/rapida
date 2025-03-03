import os
import logging
from typing import List
from pathlib import Path
import geopandas as gpd
from rich.progress import Progress
import rasterio
import rasterio.mask
import rasterio.features
from shapely.geometry import shape, mapping
from cbsurge import constants
from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.project import Project
from cbsurge.session import Session
from cbsurge.util import geo
from cbsurge.util.download_geodata import download_raster
from cbsurge.util.resolve_url import resolve_geohub_url
from cbsurge.stats.zst import zst


logger = logging.getLogger(__name__)


class RwiComponent(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # # get parent package name for electricity
        # current_dir = Path(__file__).resolve().parent
        # parent_package_name = current_dir.parents[0].name
        #
        # self.component_name = f"{parent_package_name}.{ self.__class__.__name__.lower().split('component')[0]}"

    def __call__(self, variables: List[str] = None, **kwargs) -> str:

        logger.debug(f'Assessing component "{self.component_name}" ')
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if not var_name in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        with Session() as ses:
            variables_data = ses.get_component(self.component_name)

            for var_name in variables:
                var_data = variables_data[var_name]
                var_data['source'] = resolve_geohub_url(var_data['source'])

                # create instance
                v = RwiVariable(name=var_name, component=self.component_name, **var_data)
                # assess
                v(**kwargs)


class RwiVariable(Variable):

    @property
    def affected_path(self):
        path, file_name = os.path.split(self.local_path)
        fname, ext = os.path.splitext(file_name)
        return os.path.join(path, f'{fname}_affected{ext}')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        output_filename = f"{self.component}.tif"
        self.local_path = os.path.join(os.path.dirname(geopackage_path), output_filename)

    def download(self, **kwargs):
        """
        Download RWI data source
        """
        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        cog_url = self.source

        force_compute = kwargs.get('force_compute', False)

        if force_compute == True or not os.path.exists(self.local_path):
            self.local_path = download_raster(
                dataset_url=cog_url,
                geopackage_path=geopackage_path,
                output_filename=os.path.basename(self.local_path),
                progress=kwargs.get('progress', None)
            )

        if project.raster_mask is not None and geo.is_raster(self.local_path):
            affected_local_path = self.affected_path
            if force_compute == True or not os.path.exists(affected_local_path):
                self.clip_raster_with_mask(
                    source=self.local_path,
                    mask=project.raster_mask,
                    output_path=affected_local_path
                )

    def compute(self, **kwargs):
        pass

    def resolve(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        """
        make zonal statistics for variable. compute affected variable if project has mask layer.
        """
        progress: Progress = kwargs.get('progress', Progress())
        evaluate_task = None
        if progress is not None:
            evaluate_task = progress.add_task(
                description=f'[red]Going to evaluate {self.name} in {self.component} component', total=None)

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
                progress.update(evaluate_task, description=f'Evaluating variable {self.name} using zonal stats')

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
                progress.update(evaluate_task, description=f'Evaluated variable {self.name} using zonal stats')

            if progress is not None and evaluate_task is not None:
                progress.update(evaluate_task,
                                description=f'Writing {self.name} to {project.geopackage_file_path}:{dst_layer}')

            gdf.to_file(project.geopackage_file_path, layer=dst_layer, driver="GPKG")
        else:
            progress.update(evaluate_task,
                            description=f'{self.name} was skipped because of lack of operator definition.')

        if progress and evaluate_task:
            progress.remove_task(evaluate_task)

    def clip_raster_with_mask(self, source, mask, output_path):
        """
        Clips the raster data at 'source' using the mask raster at 'mask' and saves the result to 'output_path'.

        :param source: Path to the source raster file.
        :param mask: Path to the mask raster file (e.g., project.raster_mask).
        :param output_path: File path to save the clipped raster.
        """
        # Open the mask raster and read the first band
        with rasterio.open(mask) as mask_src:
            mask_data = mask_src.read(1)
            # Define the mask region as areas where the value is greater than 0 (adjust condition as needed)
            binary_mask = mask_data > 0

            # Extract polygons from the mask raster based on the binary_mask
            mask_shapes = []
            for geom, value in rasterio.features.shapes(mask_data, mask=binary_mask, transform=mask_src.transform):
                # If the value equals 1, use that region as a mask
                if value == 1:
                    mask_shapes.append(shape(geom))

        # Open the main raster and clip it using the extracted polygons
        with rasterio.open(source) as src:
            # Convert the polygon geometries to dict format using mapping()
            out_image, out_transform = rasterio.mask.mask(src, [mapping(geom) for geom in mask_shapes], crop=True)
            out_meta = src.meta.copy()

        # Update the metadata
        out_meta.update({
            "driver": "COG",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": 'zstd',
            "predictor": 2
        })

        # Save the clipped raster to the output file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)