import logging
import sys

import dotenv
import hydra
from loguru import logger
from omegaconf import DictConfig

from zotero_arxiv_daily.inbound_email import EmailRequestProcessor


dotenv.load_dotenv()


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(config: DictConfig):
    log_level = "DEBUG" if config.executor.debug else "INFO"
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )
    for logger_name in logging.root.manager.loggerDict:
        if "zotero_arxiv_daily" not in logger_name:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    EmailRequestProcessor(config).run_once()


if __name__ == "__main__":
    main()
