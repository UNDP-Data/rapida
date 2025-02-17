import io
import json
import logging
import multiprocessing
import os
import subprocess
import tempfile
import typing
from traceback import print_exc
from osgeo import gdal, ogr, osr
from pmtiles.reader import Reader, MmapSource

from cbsurge.az.blobstorage import upload_blob
from cbsurge.session import Session
from cbsurge.util import setup_logger

logger = logging.getLogger(__name__)
gdal.UseExceptions()

ATTRIBUTION = "United Nations Development Programme (UNDP)"


def should_reproject(src_srs: osr.SpatialReference = None, dst_srs: osr.SpatialReference = None):
    """
    Decides if two projections are equal
    @param src_srs:  the source projection
    @param dst_srs: the dst projection
    @return: bool, True if the source  is different then dst else false
    If the src is ESPG:4326 or EPSG:3857  returns  False
    """
    auth_code_func_name = ".".join(
        [osr.SpatialReference.GetAuthorityCode.__module__, osr.SpatialReference.GetAuthorityCode.__name__])
    is_same_func_name = ".".join([osr.SpatialReference.IsSame.__module__, osr.SpatialReference.IsSame.__name__])
    if src_srs.GetAuthorityCode(None) and int(src_srs.GetAuthorityCode(None)) == 4326: return False
    try:

        proj_are_equal = int(src_srs.GetAuthorityCode(None)) == int(dst_srs.GetAuthorityCode(None))
    except Exception as evpe:
        logger.error(
            f'Failed to compare src and dst projections using {auth_code_func_name}. Trying using {is_same_func_name}')
        try:
            proj_are_equal = bool(src_srs.IsSame(dst_srs))
        except Exception as evpe1:
            logger.error(
                f'Failed to compare src and dst projections using {is_same_func_name}. Error is \n {evpe1}')
            raise evpe1

    return not proj_are_equal


def tippecanoe(tippecanoe_cmd: str = None, timeout_event=None):
    """
    tippecanoe is a bit peculiar. It redirects the status and live logging to stderr
    see https://github.com/mapbox/tippecanoe/issues/874

    As a result the line buffering  has to be enabled (bufsize=1) and the output is set as text (universal_new_line)
    This allows to follow the conversion logs in real time.
    @param tippecanoe_cmd: str, the
    @param timeout_event:
    @return:
    """
    logger.debug(' '.join(tippecanoe_cmd))
    with subprocess.Popen(tippecanoe_cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          start_new_session=True,
                          universal_newlines=True,
                          bufsize=1
                          ) as proc:
        # the error is going to show up on stdout as it is redirected in the Popen
        err = None
        with proc.stdout:
            stream = io.open(proc.stdout.fileno())  # this will really make it streamabale
            while proc.poll() is None:
                output = stream.readline().strip('\r').strip('\n')
                if output:
                    logger.debug(output)
                    if err != output: err = output
                if timeout_event and timeout_event.is_set():
                    logger.error(f'tippecanoe process has been signalled to stop ')
                    proc.terminate()
                    raise subprocess.TimeoutExpired(cmd=tippecanoe_cmd, timeout=None)

        if proc.returncode and proc.returncode != 0:
            raise Exception(err)


def dataset2fgb(fgb_dir: str = None,
                src_ds: typing.Union[gdal.Dataset, ogr.DataSource] = None,
                layers: typing.List[str] = None,
                dst_prj_epsg: int = 4326,
                conn_string: str = None,
                blob_url: str = None,
                timeout_event=None):
    """
    Convert one or more layers from src_ds into FlatGeobuf format in a (temporary) directory featuring dst_prj_epsg
    projection. The layer is possibly reprojected. In case errors are encountered an error blob is uploaded for now
    #TODO
    @param fgb_dir: the abs path to a directory where the FGB files will be created
    @param src_ds: GDAL Dataset  or OGR Datasource instance where the layers will be read from
    @param layers: list of layer name ot be converted
    @param dst_prj_epsg: the  target projection as an EPSG code
    @param conn_string: the connection string used to connect to the Azure storage account
    @param blob_url: the url of the blob to be ingested
    @param timeout_event:
    @return:
    """
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(dst_prj_epsg)
    src_path = os.path.abspath(src_ds.GetDescription())
    converted_layers = dict()
    for lname in layers:
        try:
            # if '_' in lname:raise Exception(f'Simulated exception on {lname}')
            dst_path = os.path.join(fgb_dir, f'{lname}.fgb')
            layer = src_ds.GetLayerByName(lname)
            original_features = layer.GetFeatureCount()
            layer_srs = layer.GetSpatialRef()

            if layer_srs is None:
                logger.error(f'Layer {lname} does not feature a projection and will not be ingested')
                continue

            fgb_opts = [
                '-f FlatGeobuf',
                '-preserve_fid',
                '-skipfailures',
                '-nlt PROMOTE_TO_MULTI',
                '-makevalid'

            ]

            reproject = should_reproject(src_srs=layer_srs, dst_srs=dst_srs)
            if reproject:
                fgb_opts.append(f'-t_srs EPSG:{dst_prj_epsg}')
            fgb_opts.append(f'"{lname}"')
            logger.debug(f'Converting {lname} from {src_path} into {dst_path}')
            logger.debug(f'srs:{layer_srs} should repr {reproject} {" ".join(fgb_opts)}')
            fgb_ds = gdal.VectorTranslate(destNameOrDestDS=dst_path,
                                          srcDS=src_ds,
                                          reproject=reproject,
                                          options=' '.join(fgb_opts),
                                          callback=gdal_callback,
                                          callback_data=timeout_event
                                          )
            converted_features = fgb_ds.GetLayerByName(lname).GetFeatureCount()
            if converted_features > 1e6:
                logger.info(f'Layer "{lname}" is quite large: {converted_features} features. Processing time can be over 30 min. ')
            logger.debug(f'Original no of features {original_features} vs converted {converted_features}')
            logger.debug(gdal.VectorInfo(fgb_ds, format='json', options='-al -so'))
            logger.info(f'Converted {lname} from {src_path} into {dst_path}')
            converted_layers[lname] = dst_path
            del fgb_ds
            #issue a warning in case the out features are 0 or there is
            if converted_features == 0 or converted_features!= original_features and conn_string:
                error_message = f'There could be issues with layer "{lname}".\nOriginal number of features/geometries ={original_features} while converted={converted_features}'
                logger.error(error_message)


        except (RuntimeError, Exception) as re:
            if 'user terminated' in str(re):
                logger.info(f'Conversion of {lname} from {src_path} to FlatGeobuf has timed out')
            else:
                with io.StringIO() as m:
                    print_exc(
                        file=m
                    )  # exc is extracted using system.exc_info
                    error_message = m.getvalue()
                    dataset_path = blob_url
                    msg = f'dataset: {dataset_path}\n'
                    msg += f'layer: {lname}\n'
                    msg += f'gdal_error_message: {error_message}'
                    logger.error(msg)

    return converted_layers


async def fgb2pmtiles(blob_url=None, fgb_layers: typing.Dict[str, str] = None, pmtiles_file_name: str = None,
                timeout_event=multiprocessing.Event()):
    """
    Converts all FlatGeobuf files from fgb_layers dict into PMtile format and uploads the result to Azure
    blob. Supports cancellation through event arg
    @param fgb_layers: a dict where the key is the layer name and the value is the abs path to the FlatGeobuf file
    @param pmtiles_file_name: the name of the output PMTiles file. If supplied all layers will be added to this file
    @param timeout_event: arg to signalize to Tippecanoe a timeout/interrupt
    @param conn_string: the connection string used t connect to the Azure storage account
    @return:
    """

    # fgb_dir = None
    try:
        assert pmtiles_file_name != '', f'Invalid PMtiles file name {pmtiles_file_name}'
        fgb_sources = list()
        fgb_dir = None
        for layer_name, fgb_layer_path in fgb_layers.items():
            fgb_sources.append(f'--named-layer={layer_name}:{fgb_layer_path}')
            if fgb_dir is None:
                fgb_dir, _ = os.path.split(fgb_layer_path)


        pmtiles_path = os.path.join(fgb_dir, f'{pmtiles_file_name}.pmtiles' if not '.pmtiles' in pmtiles_file_name else pmtiles_file_name)
        tippecanoe_cmd = [
            "tippecanoe",
            "-o",
            pmtiles_path,
            "--no-feature-limit",
            "-zg",
            "--simplify-only-low-zooms",
            "--detect-shared-borders",
            "--read-parallel",
            "--no-tile-size-limit",
            "--no-tile-compression",
            "--force",
            f'--name={pmtiles_file_name}',
            f'--description={",".join(list(fgb_layers.keys()))}',
            f'--attribution={ATTRIBUTION}',
        ]

        tippecanoe_cmd += fgb_sources
        tippecanoe(tippecanoe_cmd=tippecanoe_cmd, timeout_event=timeout_event)
        with open(pmtiles_path, 'r+b') as f:
            reader = Reader(MmapSource(f))
            mdict = reader.metadata()
            #TODO add error in append mode
            fcount = dict([(e['layer'], e['count']) for e in mdict["tilestats"]['layers']])
            for layer_name in fgb_layers:
                if layer_name not in fcount:
                    logger.error(f'{layer_name} is not present in {pmtiles_path} PMTiles file.')
                layer_feature_count = fcount[layer_name]
                if layer_feature_count == 0:
                    logger.error(f'{layer_name} from {pmtiles_path} PMTiles file is empty')


        logger.info(f'Created multilayer PMtiles file {pmtiles_path}')
        # upload layer_pmtiles_path to azure
        await upload_blob(src_path=pmtiles_path, dst_path=blob_url)

        # upload fgb files to azure
        for layer_name, fgb_layer_path in fgb_layers.items():
            await upload_blob(src_path=fgb_layer_path, dst_path=f"{blob_url}.{layer_name}.fgb")


    except subprocess.TimeoutExpired as te:
        logger.error(f'Conversion of layers {",".join(fgb_layers)} from {fgb_dir} has timed out.')

    except Exception as e:
        with io.StringIO() as m:
            print_exc(
                file=m
            )  # exc is extracted using system.exc_info
            error_message = m.getvalue()
            dataset_path = blob_url
            msg = f'dataset: {dataset_path}\n'
            msg += f'layers: {",".join(fgb_layers)}\n'
            msg += f'gdal_error_message: {error_message}'
            logger.error(msg)


def gdal_callback(complete, message, timeout_event):
    logger.debug(f'{complete * 100:.2f}%')
    if timeout_event and timeout_event.is_set():
        logger.info(f'GDAL received timeout signal')
        return 0

async def dataset2pmtiles(blob_url: str = None,
                    src_ds: gdal.Dataset = None,
                    layers: typing.List[str] = None,
                    pmtiles_file_name: typing.Optional[str] = None,
                    timeout_event=multiprocessing.Event()):
    """
    Converts the layer/s contained in src_ds GDAL dataset  to PMTiles and uploads them to Azure

    @param blob_url:
    @param src_ds: instance of GDAL Dataset
    @param layers: iter or layer/s name/s
    @param conn_string: Azure storage account connection string
    @param pmtiles_file_name: optional, the output PMtiles file name. If supplied all vector layers
    will ve stored in one multilayer PMTile file
    @param timeout_event: instance of multiprocessing.Event used to interrupt the processing
    @return: None

    The conversion is implemented in two stages

    1. every layer is converted into a FlatGeobuf file. A FlaGeobuf file supports only one layer.
    2. FGB files are converted to PMTiles using tippecanoe
        a) if pmtiles_file_name arg is supplied a multilayer OMTile file is created
        b) else each layer is extracted to it;s own OMTiles file

    Last, the PMTile files are uploaded to Azure

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        fgb_layers = dataset2fgb(fgb_dir=temp_dir,
                                 src_ds=src_ds,
                                 layers=layers,
                                 blob_url=blob_url,
                                 timeout_event=timeout_event)
        logger.info(fgb_layers)
        if fgb_layers:
            await fgb2pmtiles(blob_url=blob_url, fgb_layers=fgb_layers, pmtiles_file_name=pmtiles_file_name, timeout_event=timeout_event)


import asyncio
async def main():
    src_file = "/data/kigali/data/kigali.gpkg"
    with Session() as s:
        dst_file = f"az:{s.get_account_name()}:{s.get_publish_container_name()}/projects/kigali/kigali.pmtiles"

    try:
        vdataset = gdal.OpenEx(src_file, gdal.OF_VECTOR)
    except RuntimeError as ioe:
        if 'supported' in str(ioe):
            vdataset = None
        else:
            raise

    if vdataset is None:
        raise RuntimeError(f"{src_file} does not contain vector GIS data")

    logger.info(f'Opened {src_file} with {vdataset.GetDriver().ShortName} vector driver')
    nvector_layers = vdataset.GetLayerCount()
    if nvector_layers == 0:
        raise RuntimeError(f"{src_file} does not contain vector layers")
    logger.info(f'Found {nvector_layers} vector layers')

    _, file_name = os.path.split(vdataset.GetDescription())
    layer_names = [vdataset.GetLayerByIndex(i).GetName() for i in range(nvector_layers)]

    logger.info(f'Ingesting all vector layers into one multilayer PMtiles file')
    fname, *ext = file_name.split(os.extsep)

    await dataset2pmtiles(blob_url=dst_file, src_ds=vdataset, layers=layer_names,
                    pmtiles_file_name=fname)

if __name__ == "__main__":
    logger = setup_logger('rapida', level=logging.DEBUG)
    asyncio.run(main())