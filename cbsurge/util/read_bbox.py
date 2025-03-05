import datetime
import logging
import time
from pyogrio import open_arrow

logger = logging.getLogger(__name__)


def read_bbox(src_path=None, bbox=None, mask=None, batch_size=None, signal_event=None, name=None, ntries=3, progress=None):

    if progress:
        task = progress.add_task(description=f'[green]Downloading data in {name}...', start=False, total=None)
    try:
        for attempt in range(ntries):
            logger.debug(f'Attempt no {attempt} at {name}')
            try:
                with open_arrow(src_path, bbox=bbox, mask=mask, use_pyarrow=True, batch_size=batch_size, return_fids=True) as source:
                    meta, reader = source
                    logger.debug(f'Opened {src_path}')
                    batches = []
                    nb = 0
                    for b in reader :
                        if signal_event.is_set():
                            logger.info(f'Cancelling data extraction in {name}')
                            return name, meta, batches
                        if b.num_rows > 0:
                            batches.append(b)
                            nb+=b.num_rows
                            if progress:
                                progress.update(task, description=f'[green]Downloaded {nb} data in {name}', advance=nb, completed=None)
                    return name, meta, batches
            except Exception as e:
                if attempt < ntries-1:
                    logger.info(f'Attempting to download {name} again')
                    time.sleep(1)
                    continue
                else:
                    return name, e, None
    finally:
        if progress:
            progress.remove_task(task)

def stream(src_path=None, src_layer=0, bbox=None, mask=None, batch_size=None,
         signal_event=None, name=None, ntries=3, progress=None, results=None, add_polyid=False):
    """
    Stream geospatial vector data using leveraging pyogrio/pyarrow API

    :param src_path: str, url to stream
    :param src_layer: str or int, default=0, the layer name or layer id to read from source
    :param bbox: iterable of 4 floats (left, bottom, right top) representing a bounding box to limit the streaming in
                 same projection like src_layer
    :param mask: a shapely geometry in the same projection like src_layer
    :param batch_size: int, the number of features to stream in one batch
    :param signal_event: multiprocessing even instance to stop/cancel the streaming
    :param name: the name assigned to the mask/bbox, used to logging purposes
    :param ntries: int=3, how many times should the stream be restarted in case an error is encountyered
    :param progress:instance of  roch progress bar
    :param results: instance of collections.deque used to place the downloaded features into the main thread
    :return: str, the name of the mask/bbox
    NB. The function shuts itself down if no features are downloaded/read in 1800 seconds

    """

    try:
        for attempt in range(ntries):
            logger.debug(f'Attempt no {attempt} at {name}')
            try:
                if progress:
                    task = progress.add_task(description=f'[green]Downloading data in {name}...', start=False,
                                             total=None, )
                with open_arrow(src_path, layer=src_layer, bbox=bbox, mask=mask, use_pyarrow=True, batch_size=batch_size, return_fids=True) as source:
                    meta, reader = source
                    logger.debug(f'Opened {src_path}')
                    nb = 0
                    n = 0
                    start = datetime.datetime.now()
                    for b in reader :
                        if signal_event.is_set():
                            logger.info(f'Cancelling download in {name}')
                            return name
                        if b.num_rows > 0:
                            if add_polyid:
                                b = b.append_column('polyid', name)
                            results.append(b)
                            nb+=b.num_rows
                            if progress:progress.update(task, description=f'[green]Downloaded {nb} features in {name}',
                                                        advance=nb, completed=None, bar_style="green")

                            return name
                        now = datetime.datetime.now()
                        delta = now-start
                        if delta.total_seconds() > 1800 and n == nb:
                            raise Exception('No features downloaded in 30 minutes over {name}')
                        n = nb
                    return name
            except Exception as e:
                if attempt < ntries-1:
                    time.sleep(1)
                    continue
                else:
                    raise e
    finally:
        if progress:
            progress.remove_task(task)