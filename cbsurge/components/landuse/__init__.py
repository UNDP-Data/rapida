import logging
import os
from typing import List
from rich.progress import Progress
from cbsurge.components.landuse.stac import STAC_MAP, interpolate_stac_source, download_stac
from cbsurge.core.component import Component
from cbsurge.core.variable import Variable
from cbsurge.project import Project
from cbsurge.session import Session


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


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


    def download(self, force_compute=False, **kwargs):
        project = Project(os.getcwd())
        progress: Progress = kwargs.get('progress', None)

        output_dir = os.path.join(os.path.dirname(project.geopackage_file_path), self.component)

        asset_files = self.downloaded_files
        exists = 0
        for asset in asset_files:
            if os.path.exists(asset):
                exists += 1
        if force_compute == False and exists == len(asset_files):
            # if all files already exist, skip download
            pass
        else:
            download_stac(stac_url=self.stac_url,
                          collection_id=self.collection_id,
                          geopackage_file_path=project.geopackage_file_path,
                          polygons_layer_name=project.polygons_layer_name,
                          output_dir=output_dir,
                          target_year=self.target_year,
                          target_assets=self.target_asset,
                          target_srs=project.target_srs,
                          progress=progress)


    def _compute_affected_(self, **kwargs):
        # TODO: compute affected data for each band
        pass

    def compute(self, **kwargs):
        # TODO: Predict land use by using model
        # Then, compute affected area for land use
        self._compute_affected_(**kwargs)

    def evaluate(self, **kwargs):
        # TODO: compute zonal statistics by using land use data
        pass

    def resolve(self, **kwargs):
        pass

