import logging

logger = logging.getLogger(__name__)

def in_notebook():
    """check if code is being executed in a jupyter notebook"""
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True