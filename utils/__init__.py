import logging

from rich.logging import RichHandler


def get_logger(name, level=logging.NOTSET, **rich_kwargs):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(**rich_kwargs)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_loglevel(level):
    logging.getLogger().setLevel(level)
