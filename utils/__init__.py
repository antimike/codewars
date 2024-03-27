import logging

from rich.logging import RichHandler


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler()
        logger.addHandler(handler)
    return logger


def set_loglevel(level):
    logging.getLogger().setLevel(level)
