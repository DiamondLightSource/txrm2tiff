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

    date_format = "%H:%M:%S %d-%m-%Y"
    log_format = "%(levelname)8s: %(message)s"
    logging_format = logging.Formatter(log_format, date_format)

    logger = logging.getLogger()
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging_format)

    logger.addHandler(handler)

