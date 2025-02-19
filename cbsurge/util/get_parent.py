import logging
import os

logger = logging.getLogger(__name__)

def get_parent(directory: str) -> str:
    """Returns the parent directory of the given folder."""
    return os.path.abspath(os.path.join(directory, os.pardir))