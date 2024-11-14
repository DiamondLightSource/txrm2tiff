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


def create_logger(input_level: str | int) -> None:
    input_level = str(input_level).strip().lower()
    try:
        level = level_dict[input_level]
    except KeyError:
        logging.error(
            'Failed to parse logging level %s, "debug" will be used', input_level
        )
        level = logging.DEBUG

    date_format = "%H:%M:%S %d-%m-%Y"
    log_format = "%(levelname)8s: %(message)s"
    logging_format = logging.Formatter(log_format, date_format)

    logger = logging.getLogger("txrm2tiff")
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging_format)

    logger.addHandler(handler)
