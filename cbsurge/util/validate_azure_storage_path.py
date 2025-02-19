import logging

logger = logging.getLogger(__name__)

def validate_azure_storage_path(a_path:str|None = None):
    assert a_path.startswith(
        'az:'), f'The source blob path {a_path} is not in the correct format: az:account_name:blob_path'
    assert a_path.count(
        ':') == 2, f'The source blob path {a_path} is not in the correct format: az:account_name:blob_path'