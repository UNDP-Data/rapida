import logging
import threading

logger = logging.getLogger(__name__)

def downloader(work=None, result=None, finished=None):
    logger.debug(f'starting downloader thread {threading.current_thread().name}')
    while True:
        job = None
        try:
            job = work.pop()
        except IndexError as ie:
            pass
        if job is None:
            if finished.is_set():
                logger.debug(f'worker is finishing  in {threading.current_thread().name}')
                break
            continue

        if finished.is_set():
            break
        logger.debug(f'Starting job  {job["name"]}')
        result.append(read_bbox(**job))