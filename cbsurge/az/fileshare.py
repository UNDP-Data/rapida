import os
from collections import deque
from azure.storage.fileshare import ShareClient
import logging
import click
from cbsurge.session import Session
from cbsurge.util.setup_logger import setup_logger

logger = logging.getLogger(__name__)


def list_projects():
    """
    Yields the first level directory names in the default file share set through rapida init
    :return: name of the folders
    """
    with Session() as session:
        share_name = session.get_file_share_name()
        account_url = session.get_file_share_account_url()
        with ShareClient(account_url=account_url,share_name=share_name,
                         credential=session.get_credential(), token_intent='backup') as sc:
            for entry in sc.list_directories_and_files():
                if entry.is_directory:
                    yield entry.name



def download_project(name:str=None, dst_folder=None, progress=None, overwrite=False, max_concurrency=8):
    """
    Download a folder representing a rapida project  from the azure account and file share set through rapida init
    :param name: str, the project/fodler name to download
    :param dst_folder: local folder where the project folder(whole) will be downloaded
    :param progress: optional, instance of rich.Progress to report download status
   :param overwrite: bool, default = false, whether the files in the local path should be overwritten in case they exist
    :param max_concurrency: int, the number of threads to use in low level azure api when downloading
    :return: None
    """
    tasks = deque()
    def download_recursive(directory_client, local_path):
        """ Recursively download files and directories """
        os.makedirs(local_path, exist_ok=True)

        for item in directory_client.list_directories_and_files():
            item_name = item.name
            item_path = os.path.join(local_path, item_name)

            if not item.is_directory:  # It's a file
                file_client = directory_client.get_file_client(item_name)
                rel_path = os.path.join(directory_client.directory_path, item.name)
                stream = file_client.download_file(max_concurrency=max_concurrency)
                if os.path.exists(item_path) and overwrite is False:
                    raise FileExistsError(f'{item_path} already exists. Set overwrite=True to overwrite')
                with open(item_path, "wb") as dst:
                    if progress is not None:
                        task = progress.add_task(f"Downloading {rel_path}", total=item.size)
                        tasks.append(task)
                        for chunk in stream.chunks():
                            dst.write(chunk)
                            progress.update(task, advance=len(chunk), description=f'Downloaded {rel_path}')
                        if len(tasks) > 1:
                            progress.remove_task(tasks.popleft())
                    else:
                            stream = file_client.download_file(max_concurrency=max_concurrency)
                            dst.write(stream.readall())

            else:  # It's a directory
                sub_dir_client = directory_client.get_subdirectory_client(item_name)
                download_recursive(sub_dir_client, item_path)

    with Session() as session:
        account_name = session.get_account_name()
        share_name = session.get_file_share_name()
        account_url = f'https://{account_name}.file.core.windows.net'
        with ShareClient(account_url=account_url,share_name=share_name,
                         credential=session.get_credential(), token_intent='backup') as sc:
            with sc.get_directory_client(name) as project_dir_client:
                download_recursive(project_dir_client, dst_folder)

    if tasks and progress:
       for _ in range(len(tasks)):
            progress.remove_task(tasks.pop())


if __name__ == '__main__':
    logger = setup_logger(name='rapida')

    from rich.progress import Progress
    src_folder = '/home/work/py/geo-cb-surge/ap'
    with Progress() as p:
        #upload_project(project_folder=src_folder, overwrite=True, progress=p)
        # for e in list_projects():
        #     print(e)
        download_project(name='ap', dst_folder='/tmp', progress=p, overwrite=True)

