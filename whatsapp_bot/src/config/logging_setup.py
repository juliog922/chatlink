import logging
import os


def setup_logging():
    """
    Configure structured JSON logging using LOG_LEVEL from environment.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format=(
            '{"timestamp": "%(asctime)s", '
            '"level": "%(levelname)s", '
            '"message": "%(message)s", '
            '"module": "%(module)s"}'
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logging.getLogger("grpc").setLevel(logging.WARNING)
    logging.info("Logging initialized")
