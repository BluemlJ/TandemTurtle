"""
Log files are saved to the log folder inside the run folder.
To turn on logging, set the values of the logger_disabled variables to False inside this file.
Viewing the log files will help you to understand how the algorithm works and see inside its ‘mind’.
"""

import logging
from config import run_folder


def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


### SET all LOGGER_DISABLED to True to disable logging
### WARNING: the mcts log file gets big quite quickly

LOGGER_DISABLED = {
'main':False
, 'memory':False
, 'tourney':False
, 'mcts':False
, 'model': False}


logger_mcts = setup_logger('logger_mcts', run_folder + 'logs/logger_mcts.log')
logger_mcts.disabled = LOGGER_DISABLED['mcts']

logger_main = setup_logger('logger_main', run_folder + 'logs/logger_main.log')
logger_main.disabled = LOGGER_DISABLED['main']

logger_tourney = setup_logger('logger_tourney', run_folder + 'logs/logger_tourney.log')
logger_tourney.disabled = LOGGER_DISABLED['tourney']

logger_memory = setup_logger('logger_memory', run_folder + 'logs/logger_memory.log')
logger_memory.disabled = LOGGER_DISABLED['memory']

logger_model = setup_logger('logger_model', run_folder + 'logs/logger_model.log')
logger_model.disabled = LOGGER_DISABLED['model']
