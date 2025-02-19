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