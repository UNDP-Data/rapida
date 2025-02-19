import logging

logger = logging.getLogger(__name__)


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