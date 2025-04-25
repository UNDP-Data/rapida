import logging


logger = logging.getLogger(__name__)
def silence_httpx_az():
    #azlogger = logging.getLogger('az.core.pipeline.policies.http_logging_policy')
    azlogger = logging.getLogger('azure')
    azlogger.setLevel(logging.WARNING)
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.WARNING)
    requests_logger = logging.getLogger('requests')
    requests_logger.setLevel(logging.WARNING)
    msal_logger = logging.getLogger('msal')
    msal_logger.setLevel(logging.WARNING)
    pyogrio_logger = logging.getLogger('pyogrio')
    pyogrio_logger.setLevel(logging.WARNING)