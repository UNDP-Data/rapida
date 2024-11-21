import os.path
from urllib.parse import urljoin
from overturemaps import core, cli
from azure.storage.blob import BlobServiceClient
#https://overturemapswestus2.blob.core.windows.net/release/{release}/theme={theme}/type={ptype}
#https://overturemapswestus2.dfs.core.windows.net/release/2024-11-13.0/
#https://github.com/OSGeo/gdal/blob/master/autotest/ogr/ogr_parquet.py#L3359
OVERTURE_AZURE_STORAGE_ACCOUNT='https://overturemapswestus2.blob.core.windows.net'
OVERTURE_AZURE_CONTAINER='release'
type_theme_map = {
    "address": "addresses",
    "building": "buildings",
    "building_part": "buildings",
    "division": "divisions",
    "division_area": "divisions",
    "division_boundary": "divisions",
    "place": "places",
    "segment": "transportation",
    "connector": "transportation",
    "infrastructure": "base",
    "land": "base",
    "land_cover": "base",
    "land_use": "base",
    "water": "base",
}

def _dataset_path(base_url=None, overture_type: str = None) -> str:
    """
    Returns the s3 path of the Overture dataset to use. This assumes overture_type has
    been validated, e.g. by the CLI

    """
    # Map of sub-partition "type" to parent partition "theme" for forming the
    # complete s3 path. Could be discovered by reading from the top-level s3
    # location but this allows to only read the files in the necessary partition.
    theme = type_theme_map[overture_type]
    return f"PARQUET:{os.path.join(base_url,f'theme={theme}', f'type={overture_type}')}"

def fetch_admin(west=None, south=None, east=None, north=None, admin_level=None, clip=False, h3id_precision=7,
                overture_type='division_area'):


    pass
    #print(core._dataset_path('division_area'))

    blob_service_client = BlobServiceClient(account_url=OVERTURE_AZURE_STORAGE_ACCOUNT)
    container_client = blob_service_client.get_container_client(OVERTURE_AZURE_CONTAINER)
    release = None
    for blob in container_client.walk_blobs( delimiter='/'):
        if blob.name.startswith('_'):continue
        release = blob.name
    url = f'{os.path.join(OVERTURE_AZURE_STORAGE_ACCOUNT, OVERTURE_AZURE_CONTAINER, release)}'
    print(url)
    admin_dataset_url = _dataset_path(base_url=url,overture_type=overture_type)
    print(admin_dataset_url)

if __name__  == '__main__':
    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    # bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    west, south, east, north = bbox
    print(core.get_all_overture_types())
    fetch_admin(west=west, south=south, east=east, north=north, admin_level=1, )

