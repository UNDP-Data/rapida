from cbsurge.exposure.builtenv.buildings.fgbgdal import get_countries_for_bbox_osm, GMOSM_BUILDINGS_ROOT
from pyogrio.raw import open_arrow, write_arrow
import logging
import time
import tempfile
import os
from tqdm import tqdm
from cbsurge.util import generator_length
logger = logging.getLogger(__name__)


from osgeo import ogr, osr
srs_prj = osr.SpatialReference()
srs_prj.ImportFromEPSG(4326)
def download1(bbox=None, out_path=None, in_batches=True):
    nbuildings = 0
    with ogr.GetDriverByName('FlatGeobuf').CreateDataSource(out_path) as dst_ds:
        dst_lyr = dst_ds.CreateLayer('buildings', geom_type=ogr.wkbPolygon, srs=srs_prj)
        for country in get_countries_for_bbox_osm(bbox=bbox):
            remote_country_fgb_url = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={country}/{country}.fgb'
            # local_country_fgb_url = os.path.join(tmp_dir, f'buildings_{country}.fgb')
            if in_batches:
                with open_arrow(remote_country_fgb_url, bbox=bbox, use_pyarrow=True, batch_size=1000) as source:
                    meta, reader = source
                    fields = meta.pop('fields')
                    crs = meta['crs']
                    for field_name, field_type in  zip(reader.schema.names,reader.schema.types):
                        if field_type == 'string':
                            ogrft = ogr.OFTString
                        if field_type == 'double':
                            ogrft = ogr.OFTReal
                        if field_type == 'int64':
                            ogrft = ogr.OFTInteger64
                        if 'geometry' in field_name or 'wkb' in field_name:continue
                        dst_lyr.CreateField(ogr.FieldDefn(field_name, ogrft))

                    # non_empty_batches = (b for b in reader if b.num_rows > 0)
                    # no_batches, batches = generator_length(non_empty_batches)
                    bar_format = "{desc}: {n_fmt} buildings... {postfix}"
                    with tqdm(total=None, desc=f'Downloaded', bar_format=bar_format) as pbar:
                        for batch in reader:
                            logger.debug(f'Writing {batch.num_rows} records')
                            pbar.set_postfix_str(f'from {country}...', refresh=True)
                            #write_arrow(batch, out_path,layer='buildings',driver='FlatGeobuf',append=True, metadata=meta)
                            dst_lyr.WritePyArrow(batch)
                            nbuildings+=batch.num_rows
                            pbar.update(batch.num_rows)

                        pbar.set_postfix_str('Finished')
            else:
                with open_arrow(remote_country_fgb_url, bbox=bbox, use_pyarrow=False) as source:
                    meta, reader = source
                    write_arrow(reader, out_path,layer='buildings',driver='FlatGeobuf',append=True, metadata=meta)


if __name__ == '__main__':
    import asyncio
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(100)
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    nf = 5829
    bbox = 33.681335, -0.131836, 35.966492, 1.158979  # KEN/UGA
    #bbox = 19.5128619671,40.9857135911,19.5464217663,41.0120783699  # ALB, Divjake
    # bbox = 31.442871,18.062312,42.714844,24.196869 # EGY/SDN
    # bbox = 15.034157,49.282809,16.02842,49.66207 # CZE
    bbox = 19.350384,41.206737,20.059003,41.571459 # ALB, TIRANA
    bbox = 19.726666,39.312705,20.627545,39.869353, # ALB/GRC

    #a =  asyncio.run(get_admin_level_bbox(iso3='ZWE'))
    #print(a)
    url = 'https://undpgeohub.blob.core.windows.net/userdata/9426cffc00b069908b2868935d1f3e90/datasets/bldgsc_20241029084831.fgb/bldgsc.pmtiles?sv=2025-01-05&ss=b&srt=o&se=2025-11-29T15%3A58%3A37Z&sp=r&sig=bQ8pXRRkNqdsJbxcIZ1S596u4ZvFwmQF3TJURt3jSP0%3D'
    #validate_source()
    out_path = '/tmp/bldgs1.fgb'
    # cntry = get_countries_for_bbox_osm(bbox=bbox)
    # print(cntry)
    start = time.time()
    #asyncio.run(download(bbox=bbox))

    download1(bbox=bbox, out_path=out_path)

    end = time.time()
    print((end-start))