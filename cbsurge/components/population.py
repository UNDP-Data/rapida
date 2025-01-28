import asyncio
from typing import List
from cbsurge.components.component_base import ComponentBase
from cbsurge.exposure.population import run_download


class Population(ComponentBase):

    def __init__(self):
        self.available_variables = [
            {
                "id": "total_population",
                "title": "Total Population",
                "file_name": "{iso3}_total.tif",
            },
            {
                "id": "male_population",
                "title": "Male Population",
                "file_name": "{iso3}_male_total.tif",
            },
            {
                "id": "female_population",
                "title": "Female Population",
                "file_name": "{iso3}_female_total.tif",
            },
            {
                "id": "active_population",
                "title": "Active Population",
                "file_name": "{iso3}_active_total.tif",
            },
            {
                "id": "male_active_population",
                "title": "Male Active Population",
                "file_name": "{iso3}_male_active.tif",
            },
            {
                "id": "female_active_population",
                "title": "Female Active Population",
                "file_name": "{iso3}_female_active.tif",
            },
            {
                "id": "child_population",
                "title": "Child Population",
                "file_name": "{iso3}_child_total.tif",
            },
            {
                "id": "male_child_population",
                "title": "Male Child Population",
                "file_name": "{iso3}_male_child_total.tif",
            },
            {
                "id": "female_child_population",
                "title": "Female Child Population",
                "file_name": "{iso3}_female_child_total.tif",
            },
            {
                "id": "elderly_population",
                "title": "Elderly Population",
                "file_name": "{iso3}_elderly_total.tif",
            },
            {
                "id": "male_elderly_population",
                "title": "Male Elderly Population",
                "file_name": "{iso3}_male_elderly_total.tif",
            },
            {
                "id": "female_elderly_population",
                "title": "Female Elderly Population",
                "file_name": "{iso3}_female_elderly_total.tif",
            },
            {
                "id": "dependency",
                "title": "Dependency",
                "formula": ["child_population", "+", "elderly_population", "/", "active_population", "*", 100]
            },
            {
                "id": "child_dependency",
                "title": "Child Dependency",
                "formula": ["child_population", "/", "active_population", "*", 100]
            },
            {
                "id": "elderly_dependency",
                "title": "Elderly Dependency",
                "formula": ["elderly_population", "/", "active_population", "*", 100]
            },
            {
                "id": "population_rate_below_5_years",
                "title": "Population Rate Below 5 Years",
                "formula": ["child_population", "/", "total_population", "*", 100]
            },
            {
                "id": "population_rate_above_65_years",
                "title": "Population Rate Above 65 Years",
                "formula": ["elderly_population", "/", "total_population", "*", 100]
            },
        ]

    def find_variable_definition(self, id: str):
        variable = None
        for v in self.available_variables:
            if v["id"] == id:
                variable = v
            else:
                continue
        return variable

    def get_available_variables(self) -> List[str]:
        return [variable["id"] for variable in self.available_variables]

    async def download(self, output_folder: str, bbox: List[float] = None, countries: List[str] = None,
                 variables: List[str] = None) -> List[str]:
        if bbox is None and countries is None:
            raise ValueError("Either bbox or countries must be provided.")
        if bbox is not None and countries is not None:
            raise ValueError("Both bbox and countries cannot be used at the same time.")
        if countries is not None and len(countries) == 0:
            raise ValueError("At least a country must be provided.")
        if bbox is not None and len(bbox) != 4:
            raise ValueError("bbox must be a list of four numeric values (min longitude, min latitude, max longitude, max latitude).")

        available_variables = self.get_available_variables()
        target_variables = []
        if variables is not None:
            for vid in variables:
                if vid not in available_variables:
                    raise ValueError(f"Variable {vid} is not supported.")
                target_variables.append(vid)
        if len(target_variables) == 0:
            target_variables = available_variables

        if bbox is not None:
            # TODO: convert bbox to country iso codes
            pass

        downloaded_files = []
        for country in countries:
            files = await run_download(country_code=country, download_path=output_folder)
            downloaded_files = downloaded_files + files
        return downloaded_files

    async def assess(self, admin_file: str, output_folder: str, mask_file: str = None, masked_value: float = None,
               variables: List[str] = None) -> str:
        pass


if __name__ == "__main__":
    pop = Population()
    ids = pop.get_available_variables()
    print(ids)

    dist_folder = "/data/pop"
    # asyncio.run(pop.download(output_folder=dist_folder))
    asyncio.run(pop.download(output_folder=dist_folder, countries=["RWA"]))
    # asyncio.run(pop.download(output_folder=dist_folder, bbox=[28.657730,-2.927526,30.953872,-0.963147]))
    # asyncio.run(pop.download(output_folder=dist_folder, countries=["RWA"], bbox=[28.657730, -2.927526, 30.953872, -0.963147]))
