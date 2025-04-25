import os.path
from urllib.parse import urljoin

from azure.storage.blob import BlobServiceClient
from osgeo import gdal, ogr
import logging
from typing import Optional
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from overturemaps.cli import get_writer, copy
from overturemaps.core import record_batch_reader
from adlfs import AzureBlobFileSystem
from pyarrow.fs import PyFileSystem, FSSpecHandler
import itertools

logger = logging.getLogger('__name__')

#https://overturemapswestus2.blob.core.windows.net/release/{release}/theme={theme}/type={ptype}
#https://overturemapswestus2.dfs.core.windows.net/release/2024-11-13.0/
#https://github.com/OSGeo/gdal/blob/master/autotest/ogr/ogr_parquet.py#L3359
OVERTURE_AZURE_STORAGE_ACCOUNT='overturemapswestus2'
OVERTURE_AZURE_STORAGE_ACCOUNT_URL=f'https://{OVERTURE_AZURE_STORAGE_ACCOUNT}.blob.core.windows.net'
OVERTURE_AZURE_CONTAINER='release'


def generator_length(gen):
    gen1, gen2 = itertools.tee(gen)
    length = sum(1 for _ in gen1)  # Consume the duplicate
    return length, gen2  # Return the length and the unconsumed generator


def download(bbox, output_format, output, type_):
    

    reader = record_batch_reader(type_, bbox)
    if reader is None:
        return

    with get_writer(output_format, output, schema=reader.schema) as writer:
        copy(reader, writer)
#SUBTYPES = country,dependency,region,county,localadmin,locality,macrohood,neighborhood
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


def geoarrow_schema_adapter(schema: pa.Schema) -> pa.Schema:
    """
    Convert a geoarrow-compatible schema to a proper geoarrow schema

    This assumes there is a single "geometry" column with WKB formatting

    Parameters
    ----------
    schema: pa.Schema

    Returns
    -------
    pa.Schema
    A copy of the input schema with the geometry field replaced with
    a new one with the proper geoarrow ARROW:extension metadata

    """
    geometry_field_index = schema.get_field_index("geometry")
    geometry_field = schema.field(geometry_field_index)
    geoarrow_geometry_field = geometry_field.with_metadata(
        {b"ARROW:extension:name": b"geoarrow.wkb"}
    )

    geoarrow_schema = schema.set(geometry_field_index, geoarrow_geometry_field)

    return geoarrow_schema

def rec_batch_reader(path=None, bbox=None, subtype=None, account_name=OVERTURE_AZURE_STORAGE_ACCOUNT, ) -> Optional[pa.RecordBatchReader]:
    """
    Return a pyarrow RecordBatchReader for the desired bounding box and s3 path
    """
    
    filter = None
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        filter = (
            (pc.field("bbox", "xmin") < xmax)
            & (pc.field("bbox", "xmax") > xmin)
            & (pc.field("bbox", "ymin") < ymax)
            & (pc.field("bbox", "ymax") > ymin)
        )
        
    if subtype and filter is not None:
        filter &= pc.field('subtype') == subtype
    

    logger.info(f'Applying filter {filter}')
    
    adlfs = AzureBlobFileSystem(account_name=account_name,anon=True)
    pa_fs = PyFileSystem(FSSpecHandler(adlfs))
    
    
    dataset = ds.dataset(
        path, filesystem=pa_fs,format='parquet'
    )
    #columns=['id', 'country', 'version', 'subtype', 'names.primary', 'is_disputed', 'region', 'hierarchies']

    batches = dataset.to_batches(filter=filter)
    logger.info('E')
    # to_batches() can yield many batches with no rows. I've seen
    # this cause downstream crashes or other negative effects. For
    # example, the ParquetWriter will emit an empty row group for
    # each one bloating the size of a parquet file. Just omit
    # them so the RecordBatchReader only has non-empty ones. Use
    # the generator syntax so the batches are streamed out
    non_empty_batches = (b for b in batches if b.num_rows > 0)
    #num_batches, batches = generator_length(non_empty_batches)
    geoarrow_schema = geoarrow_schema_adapter(dataset.schema)
    reader = pa.RecordBatchReader.from_batches(geoarrow_schema, non_empty_batches)
    return reader



def fetch_overture_releases():
    
    with BlobServiceClient(account_url=OVERTURE_AZURE_STORAGE_ACCOUNT_URL) as blob_service_client :
        with blob_service_client.get_container_client(OVERTURE_AZURE_CONTAINER) as container_client:
            releases = list()
            for blob in container_client.walk_blobs( delimiter='/'):
                if blob.name.startswith('_'):continue
                releases.append(blob.name[:-1])
            return tuple(releases)

REALEASES = fetch_overture_releases()

def _dataset_path(overture_type: str = None, release=None) -> str:
    """
    Returns the az path of the Overture dataset to use. This assumes overture_type has
    been validated, e.g. by the CLI

    """
    # Map of sub-partition "type" to parent partition "theme" for forming the
    # complete s3 path. Could be discovered by reading from the top-level s3
    # location but this allows to only read the files in the necessary partition.
    theme = type_theme_map[overture_type]
    return f"PARQUET:{os.path.join(f'/vsiaz/release/{release}/theme={theme}', f'type={overture_type}')}//"

def fetch_admin(west=None, south=None, east=None, north=None, admin_level=None, clip=False, h3id_precision=7,
                overture_type='division_area', release=None):


    pass

    if release:
        assert release in REALEASES, f'Invalid Overture release {release}. Valid values are {','.join(REALEASES)}'
        logger.info(f'Using overture release {release}')
    else:
        release = REALEASES[-1]
        logger.info(f'Using latest overture release {release}')
    
    admin_dataset_url = _dataset_path(overture_type=overture_type, release=release)
    gdal.UseExceptions()
    with gdal.config_options({'AZURE_NO_SIGN_REQUEST': 'YES', 'AZURE_STORAGE_ACCOUNT':OVERTURE_AZURE_STORAGE_ACCOUNT}):
        logger.info(f'Reading {admin_dataset_url}')
        #result = gdal.VectorInfo(admin_dataset_url, format='json', deserialize=True, )
        
        with ogr.Open(admin_dataset_url, ) as src_ds:
            lyr = src_ds.GetLayer(0)
            lyr.SetSpatialFilterRect(west, south, east, north)
            geom_type = lyr.GetGeomType()
            with ogr.GetDriverByName('FlatGeobuf').CreateDataSource('/tmp/adminj.fgb') as dst_ds:
                dst_lyr = dst_ds.CreateLayer('adminj', geom_type=ogr.wkbMultiPolygon,)
                stream = lyr.GetArrowStream(["MAX_FEATURES_IN_BATCH=50"])
                schema = stream.GetSchema()
                for i in range(schema.GetChildrenCount()):
                    if schema.GetChild(i).GetName() != lyr.GetGeometryColumn():
                        s = schema.GetChild(i)
                        #print(s.GetName())
                        dst_lyr.CreateFieldFromArrowSchema(s)
                while True:
                    array = stream.GetNextRecordBatch()
                    if array is None:
                        break
                    assert dst_lyr.WriteArrowBatch(schema, array) == ogr.OGRERR_NONE

def get_fsspec_path(overture_type: str = None, release=None):
    theme = type_theme_map[overture_type]
    
    return f'release/{release}/theme={theme}/type={overture_type}/'
    

def fetch_admin_pa(west=None, south=None, east=None, north=None, admin_level=None, clip=False, h3id_precision=7,
                overture_type='division_area', release=None):


    if release:
        assert release in REALEASES, f'Invalid Overture release {release}. Valid values are {','.join(REALEASES)}'
        logger.info(f'Using overture release {release}')
    else:
        release = REALEASES[-1]
        logger.info(f'Using latest overture release {release}')
    
    fss_admin_path = get_fsspec_path(overture_type=overture_type, release=release)
    
    reader = rec_batch_reader( path=fss_admin_path, bbox=(west,south,east,north), subtype='region')
    print(reader)
    
    
    while True:
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break
        if batch.num_rows > 0:
            logger.info(f'writing {batch.num_rows} features')
            
    reader.close()
    
    # with ogr.Open(admin_dataset_url, ) as src_ds:
    #     lyr = src_ds.GetLayer(0)
    #     lyr.SetSpatialFilterRect(west, south, east, north)
    #     geom_type = lyr.GetGeomType()
    #     with ogr.GetDriverByName('FlatGeobuf').CreateDataSource('/tmp/adminj.fgb') as dst_ds:
    #         dst_lyr = dst_ds.CreateLayer('adminj', geom_type=geom_type,)
    #         stream = lyr.GetArrowStream(["MAX_FEATURES_IN_BATCH=5"])
    #         schema = stream.GetSchema()
    #         for i in range(schema.GetChildrenCount()):
    #             if schema.GetChild(i).GetName() != lyr.GetGeometryColumn():
    #                 s = schema.GetChild(i)
    #                 print(s.GetName())
    #                 dst_lyr.CreateFieldFromArrowSchema(s)
    #         while True:
    #             array = stream.GetNextRecordBatch()
    #             if array is None:
    #                 break
    #             assert dst_lyr.WriteArrowBatch(schema, array) == ogr.OGRERR_NONE
if __name__  == '__main__':
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    import time
    
    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    # bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    west, south, east, north = bbox
    start = time.time()
    fetch_admin(west=west, south=south, east=east, north=north, admin_level=1, )
    #download(bbox,'geoparquet', '/tmp/admino.parquet','division_area')
    end = time.time()
    logger.info(f'COmputation laster {end-start}')