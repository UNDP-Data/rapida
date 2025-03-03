import collections
import logging
import multiprocessing
import threading
from io import StringIO
import traceback
from cbsurge.util.read_bbox import stream, read_bbox

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



def worker(jobs:collections.deque[dict]=None, stop:multiprocessing.Event=None, task:int=None):
    """
    Worker that manages the streaming of geospatial data from a remote source over spatial partitions represented by
    bounding boxes or polygon geometries

    :param jobs: queue where jobs are  extracted from
    :param stop: instance of multiprocessing.Event, used to stop the worker
    :param task: int, the id of the parent rich progress task
    :return: None

    The worker runs in an infinite loop and executes download jobs for partitions placed in the queue.
    The general usage pattern is

     with concurrent.futures.ThreadPoolExecutor(max_workers=NWORKERS) as executor:
        jobs = dequeue()
        results = dequeue()
        for work in work_to_do:
            job = ...
            jobs.append(job)
        futures = [executor.submit(download_worker, jobs=jobs, task=total_task,stop=stop) for i in range(NWORKERS)]
        while True:
            pyarrow_batch_rec = results.pop()
            #process geoms (filter, reproject, etc)
            #write data to disk
            destination_layer.WritePyArrow(batch)
            destination_layer.SyncToDisk()
    """


    logger.debug(f'starting downloader thread {threading.current_thread().name}')


    while len(jobs) > 0 and not stop.is_set():
        job = jobs.pop()
        if stop.is_set():
            logger.debug(f'Worker was signalled to stop in {threading.current_thread().name}')
            break
        progress = job.get('progress', None)
        description = None
        try:
            logger.debug(f'Starting to download in  {job["name"]}')
            rname = stream(**job)
            logger.debug(f'Finished in  {rname}')
            description = f'[red]Downloaded features covering {rname}'

        except Exception as e:
            with StringIO() as eio:
                traceback.print_exception(e, file=eio)
                eio.seek(0)
                msg = eio.read()
                logger.debug(f'Failed to download data over {job["name"]}')
                logger.debug(msg)

            description = f'[red]Failed to download features covering {job["name"]}'
        finally:
            if progress and task:
                progress.update(task, description=description, advance=1)
    logger.debug(f'No more jobs. Worker running in {threading.current_thread().name} is finishing.')