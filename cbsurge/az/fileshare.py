import os
import time

from azure.storage.fileshare import ShareClient
import logging
from cbsurge import util
from cbsurge.session import Session
logger = logging.getLogger(__name__)

def upload_project(project_folder:str = None,  progress=None, overwrite=False ):
    with Session() as session:
        account_name = session.get_account_name()
        share_name = session.get_file_share_name()
        account_url = f'https://{account_name}.file.core.windows.net'
        project_name = project_folder.split(os.path.sep)[-1]
        with ShareClient(account_url=account_url,share_name=share_name, credential=session.get_credential(), token_intent='backup') as sc:
            with sc.get_directory_client(project_name) as project_dir_client:
                for root, dirs, files in os.walk(project_folder):
                    directory_name = os.path.relpath(root, project_folder)
                    dc = project_dir_client.get_subdirectory_client(directory_name=directory_name)
                    if not dc.exists():
                        dc.create_directory()
                    for name in files:
                        src_path = os.path.join(root, name)
                        sfc = dc.get_file_client(name)
                        web_path = os.path.join(account_url, share_name, dc.directory_path, name)
                        if sfc.exists and not overwrite:

                            raise FileExistsError(f'{web_path} already exists. Set overwrite=True to overwrite')
                        size = os.path.getsize(src_path)
                        with open(src_path, 'rb') as src:
                            if progress:
                                with progress.wrap_file(src,  total=size, description=f'Uploading {name} ',) as up:
                                    dc.upload_file(name, up,  max_concurrency=8)
                                    up.progress.remove_task(up.task)
                            else:
                                dc.upload_file(name, src_path, max_concurrency=8)

if __name__ == '__main__':
    logger = util.setup_logger(name='rapida')

    from rich.progress import Progress
    src_folder = '/home/work/py/geo-cb-surge/ap'
    with Progress() as p:
        upload_project(project_folder=src_folder, overwrite=True, progress=p)

