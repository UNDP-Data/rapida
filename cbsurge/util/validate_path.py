import logging
import os

logger = logging.getLogger(__name__)

def validate_path(src_path=None):
    assert os.path.isabs(src_path), f'{src_path} has to be a file'
    out_folder, file_name = os.path.split(src_path)
    assert os.path.exists(out_folder), f'Folder {src_path} has to exist'

    if os.path.exists(src_path):
        assert os.access(src_path, os.W_OK), f'Can not write to {src_path}'
