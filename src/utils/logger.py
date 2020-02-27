"""
logging stuff
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import logging
import warnings

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data.', UserWarning)

def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def add_filehandler(logger, filepath, level=logging.DEBUG):
    fh = logging.FileHandler(filepath)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
