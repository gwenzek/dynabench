# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
import sys

from tqdm import tqdm


def init_logger(name):
    logging.basicConfig(level=logging.INFO)
    os.makedirs("../logs", exist_ok=True)
    file_handler = logging.FileHandler(f"../logs/dynabench-server-{name}.log")
    formatter = logging.Formatter("%(name)s %(asctime)s %(msg)s")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    # Set logging level of other libraries
    annoying_loggers = ["boto3.resources", "botocore", "urllib3.connectionpool", "s3transfer"]
    for name in annoying_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger("builder")
