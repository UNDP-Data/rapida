import logging
import os
from osgeo import gdal


logger = logging.getLogger(__name__)
gdal.UseExceptions()


def publish_project(src_file_path: str):
    """
    Publish a vector file to Azure and GeoHub.
    """
    logger.info(f"Publishing {src_file_path}")

    if not os.path.exists(src_file_path):
        raise FileNotFoundError(f"File {src_file_path} not found")

    # handle vectors first
    logger.debug(f'Opening {src_file_path}')
    try:
        vdataset = gdal.OpenEx(src_file_path, gdal.OF_VECTOR)
    except RuntimeError as ioe:
        if 'supported' in str(ioe):
            vdataset = None
        else:
            raise

    if vdataset is None:
        raise RuntimeError(f"{src_file_path} does not contain vector GIS data")

    logger.info(f'Opened {src_file_path} with {vdataset.GetDriver().ShortName} vector driver')
    nvector_layers = vdataset.GetLayerCount()
    if nvector_layers == 0:
        raise RuntimeError(f"{src_file_path} does not contain vector layers")
    logger.info(f'Found {nvector_layers} vector layers')

    _, file_name = os.path.split(vdataset.GetDescription())
    layer_names = [vdataset.GetLayerByIndex(i).GetName() for i in range(nvector_layers)]

    logger.info(f'Ingesting all vector layers into one multilayer PMtiles file')
    fname, *ext = file_name.split(os.extsep)
    logger.info(fname)

    # dataset2pmtiles(blob_url=blob_url, src_ds=vdataset, layers=layer_names,
    #                 pmtiles_file_name=fname, timeout_event=timeout_event, conn_string=conn_string,
    #                 dst_directory=dst_directory)