import logging
import time
from pyogrio import open_arrow

logger = logging.getLogger(__name__)

def read_bbox(src_path=None, bbox=None, mask=None, batch_size=None, signal_event=None, name=None, ntries=3, progress=None):
    task = progress.add_task(description=f'[green]Downloading in {name}...', start=False, total=None)
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
                            logger.info(f'Cancelling extraction in {name}')
                            return name, meta, batches
                        if b.num_rows > 0:
                            batches.append(b)
                            nb+=b.num_rows
                            progress.update(task, description=f'[green]Downloaded {nb} in {name}', advance=nb, completed=None)
                    return name, meta, batches
            except Exception as e:
                if attempt < ntries-1:
                    logger.info(f'Attempting to download {name} again')
                    time.sleep(1)
                    continue
                else:
                    return name, e, None
    finally:
        progress.remove_task(task)