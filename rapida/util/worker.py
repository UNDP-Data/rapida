import collections
import logging
import multiprocessing
import threading
import typing
from io import StringIO
import traceback


logger = logging.getLogger(__name__)


def worker(job: typing.Callable = None, jobs: collections.deque[dict] = None, stop: multiprocessing.Event = None,
           task: int = None, id_prop_name=None):
    """
    Generic worker that manages jobs

    :param job, a func to be called with items from jobs
    :param jobs_kwargs: queue with job kwargs as dict where jobs are  extracted from
    :param stop: instance of multiprocessing.Event, used to stop the worker
    :param task: int, the id of the parent rich progress task
    :param id_prop_name, str, the name of the arg in job kwargs used to identify a single job
    :return: None

    The worker runs in an infinite loop and executes download jobs for partitions placed in the queue.
    The general usage pattern is

     with concurrent.futures.ThreadPoolExecutor(max_workers=NWORKERS) as executor:
        jobs = dequeue()
        results = dequeue()
        for work in work_to_do:
            job = ...
            jobs.append(job)
        futures = [executor.submit(worker, job=function_to_call, jobs_kwargs=jobs, task=total_task,stop=stop) for i in range(NWORKERS)]
        while True:
            pyarrow_batch_rec = results.pop()
            #process geoms (filter, reproject, etc)
            #write data to disk
            destination_layer.WritePyArrow(batch)
            destination_layer.SyncToDisk()
    """
    logger.debug(f'starting worker thread {threading.current_thread().name}')

    while len(jobs) > 0 :
        job_kwargs = jobs.pop()
        job_id = job_kwargs.get(id_prop_name or '')
        if stop.is_set():
            logger.debug(f'Worker was signalled to stop in {threading.current_thread().name}')
            break
        progress = job_kwargs.get('progress', None)

        description = None
        try:
            logger.debug(f'Starting job with {job_id}')
            rname = job(**job_kwargs)
            logger.debug(f'Finished job  {job_id}')
            # description = f'[red]Handled features covering {rname}'

        except Exception as e:
            with StringIO() as eio:
                traceback.print_exception(e, file=eio)
                eio.seek(0)
                msg = eio.read()
                logger.error(f'Failed to handle data over {job_id}')
                logger.error(msg)

            # description = f'[red]Failed to handle features covering {job_id}'
        finally:
            if progress is not None and task is not None:
                progress.update(task, description=description, advance=1)
    logger.debug(f'No more jobs. Worker running in {threading.current_thread().name} is finishing.')