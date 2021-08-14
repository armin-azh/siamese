import logging
from pathlib import Path
from settings import BASE_DIR, LOG_CONF

LOG_DIR = Path(LOG_CONF["log"]) if Path(LOG_CONF["log"]).is_absolute() else Path(BASE_DIR).joinpath(LOG_CONF["log"])
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(log_path: Path,name:str) -> logging.Logger:
    logging.basicConfig(filename=str(log_path),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
