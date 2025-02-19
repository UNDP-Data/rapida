import logging

from rich.logging import RichHandler

from cbsurge.util.fq_function_name_formatter import FQFunctionNameFormatter
from cbsurge.util.silence_httpx_az import silence_httpx_az

logger = logging.getLogger(__name__)

def setup_logger(name=None, make_root=True,  level=logging.INFO):

    if make_root:
        logger = logging.getLogger()

    else:
        logger = logging.getLogger(name)

    # formatter = logging.Formatter(
    #     "%(module)s.%(filename)s.%(funcName)s:%(lineno)d:%(levelname)s:%(message)s",
    #     "%Y-%m-%d %H:%M:%S",
    # )
    formatter = FQFunctionNameFormatter(
    "%(fqfunc)s: %(message)s",
    "%Y-%m-%d %H:%M:%S"
)
    logging_stream_handler = RichHandler(rich_tracebacks=True)
    if level == logging.DEBUG:
        logging_stream_handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(logging_stream_handler)
    logger.name = name
    silence_httpx_az()
    return logger