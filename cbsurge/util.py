import httpx
import logging
from osgeo import gdal
import itertools
import click
import os
logger = logging.getLogger(__name__)

def silence_httpx_az():
    azlogger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
    azlogger.setLevel(logging.WARNING)
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.WARNING)

def chunker(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def gen_blocks_bbox(ds=None,blockxsize=None, blockysize=None, xminc=None, yminr=None, xmaxc=None, ymaxr=None ):
    """
    Generate reading block for gdal ReadAsArray limited by a bbox
    """


    width = ds.RasterXSize
    height = ds.RasterXSize
    wi = list(range(0, width, blockxsize))
    if width % blockxsize != 0:
        wi += [width]
    hi = list(range(0, height, blockysize))
    if height % blockysize != 0:
        hi += [height]
    for col_start, col_end in zip(wi[:-1], wi[1:]):
        col_size = col_end - col_start
        if  xminc > col_end or xmaxc < col_start:continue
        if col_start < xminc:col_start = xminc
        if col_start+col_size>xmaxc:col_size=xmaxc-col_start
        for row_start, row_end in zip(hi[:-1], hi[1:]):
            if yminr > row_end or ymaxr < row_start :continue
            if row_start<yminr:row_start=yminr
            row_size = row_end - row_start
            if row_start+row_size>ymaxr:row_size= ymaxr-row_start
            yield col_start, row_start, col_size, row_size



def gen_blocks(blockxsize=None, blockysize=None, width=None, height=None ):
    """
    Generate reading block for gdal ReadAsArray
    """
    wi = list(range(0, width, blockxsize))
    if width % blockxsize != 0:
        wi += [width]
    hi = list(range(0, height, blockysize))
    if height % blockysize != 0:
        hi += [height]
    for col_start, col_end in zip(wi[:-1], wi[1:]):
        col_size = col_end - col_start
        for row_start, row_end in zip(hi[:-1], hi[1:]):
            row_size = row_end - row_start
            yield col_start, row_start, col_size, row_size

def fetch_drivers():
    d = dict()
    for i in range(gdal.GetDriverCount()):

        drv = gdal.GetDriver(i)
        d[drv.ShortName] = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
    return d

def http_get_json(url=None, timeout=None):
    """
    Generic HTTP get function using httpx
    :param url: str, the url to be fetched
    :param timeout: httpx.Timeout instance
    :return: python dict representing the result as parsed json
    """
    assert timeout is not None, f'Invalid timeout={timeout}'
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            return response.json()


def validate(url=None, timeout=10):
    """
    Generic HTTP get function using httpx
    :param url: str, the url to be fetched
    :param timeout: httpx.Timeout instance
    :return: python dict representing the result as parsed json
    """
    with httpx.Client(timeout=timeout) as client:
        response = client.head(url, timeout=timeout)
        response.raise_for_status()



def generator_length(gen):
    """
    compute the no of elems inside a generator
    :param gen:
    :return:
    """
    gen1, gen2 = itertools.tee(gen)
    length = sum(1 for _ in gen1)  # Consume the duplicate
    return length, gen2  # Return the length and the unconsumed generator


class BboxParamType(click.ParamType):
    name = "bbox"
    def convert(self, value, param, ctx):
        try:
            bbox = tuple([float(x.strip()) for x in value.split(",")])
            fail = False
        except ValueError:  # ValueError raised when passing non-numbers to float()
            fail = True

        if fail or len(bbox) != 4:
            self.fail(
                f"bbox must be 4 floating point numbers separated by commas. Got '{value}'"
            )

        return bbox

def validate_path(src_path=None):
    assert os.path.isabs(src_path), f'{src_path} has to be a file'
    out_folder, file_name = os.path.split(src_path)
    assert os.path.exists(out_folder), f'Folder {src_path} has to exist'

    if os.path.exists(src_path):
        assert os.access(src_path, os.W_OK), f'Can not write to {src_path}'