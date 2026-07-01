import json
import logging
from pathlib import Path
from rapida.components.population import constants as wpopconst
from pystac_client import Client
from osgeo import gdal


logger = logging.getLogger(__name__)
WPOP_STAC_URL = 'https://api.stac.worldpop.org/'

gdal.UseExceptions()



def buid_vrt(sources:list[str]=None, dst_path:str=None, dst_srs:str=None):
    pass


def stac_search(stac_url=WPOP_STAC_URL, bbox:tuple[float, float, float, float]=None,
                year:int=None,  sex_category:str='total', age_group:str='total',project:str="Age and Sex Structures", resolution='100m'
                ):

    logger.info(f'Searching data from {stac_url}  ')

    query = {}

    # Using the valid queryables from the schema
    if year:
        query["year"] = {"eq": f'{year}'}
    if project:
        query["project"] = {"eq": project}
    if resolution:
        query["resolution"] = {"eq": resolution}


    # Connect to the WorldPop STAC API
    client = Client.open(url=stac_url)

    search_result = client.search(bbox=bbox, query=query)

    old = [
      "male_child",
      "child_total",
      "male_active",
      "active_total",
      "male_elderly",
      "elderly_total",
      "male_total",
      "female_child",
      "female_active",
      "female_elderly",
      "female_total",
      "total",
      "dependency",
      "child_dependency",
      "elderly_dependency"
    ]
    license = "Creative Commons Attribution 4.0 International"
    attribution = "WorldPop"
    variables = dict()
    requested_variable_name = f'{sex_category}_{age_group}' if sex_category != 'total' else f'{age_group}_{sex_category}'

    for itm in search_result.items():
        # vname = f'{itm.collection_id}_{sex_category}_{age_group}' if sex_category != age_group else f'{itm.collection_id}_{sex_category}'

        for asset_key, asset in itm.assets.items():
            if not '_' in asset_key: continue
            asset_sex_category, asset_age_str, asset_year_str = asset_key.split('_')
            try:
                asset_age = int(asset_age_str)
                for asset_age_group, (age_start_range, age_end_range) in wpopconst.WORLDPOP_AGE_MAPPING.items():
                    if age_start_range <= asset_age <= age_end_range:
                        if asset_sex_category == 'total':
                            name = f'{asset_age_group}_{asset_sex_category}'
                        else:
                            name = f'{asset_sex_category}_{asset_age_group}'
                        break
            except ValueError as e:
                asset_age_group, asset_sex_cat, asset_year_str = asset_key.split('_')
                asset_sex_category = wpopconst.SEX_MAPPING[asset_sex_cat]
                name = f'{asset_sex_category}_{asset_age_group}'
            title = f'{asset_sex_category.capitalize()} {asset_age_group} population'
            if name != requested_variable_name:continue

            if not name in variables:
                variables[name] = {'source': asset.href}

            else:

                if not 'sources' in variables[name]:
                    variables[name]['sources'] = [asset.href]
                else:
                    variables[name]['sources'].append(asset.href)
                first = variables[name].pop('source', None)
                if first is not None :
                    variables[name]['sources'].append(first)
                if not title in variables[name]:
                    variables[name].update(title=title,operator = 'sum', license = license, attribution = attribution)



        # total

        break


    # print(set(old).difference(variables))
    print(json.dumps(variables, indent=2))