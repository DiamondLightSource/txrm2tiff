import logging
import sys

level_dict = {
    "1": logging.DEBUG,
    "2": logging.INFO,
    "3": logging.WARNING,
    "4": logging.ERROR,
    "5": logging.CRITICAL,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    }


def create_logger(input_level):
    try:
        level = level_dict[input_level]
    except Exception as e:
        logging.error(e)

    logging_format = logging.Formatter("%(levelname)-8s %(message)s")
    handler = logging.StreamHandler(sys.stdout)

    logger = logging.getLogger(__name__)

    logger.setLevel(level)
    handler.setFormatter(logging_format)
    logger.addHandler(handler)
