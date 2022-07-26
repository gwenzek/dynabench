# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys


def init_logger(name):
    logger = logging.getLogger()

    logger.setLevel(logging.NOTSET)

    stderr_handler = logging.StreamHandler(sys.stdout)
    stderr_handler.setLevel(logging.INFO)
    logger.addHandler(stderr_handler)

    # Set logging level of other libraries
    annoying_loggers = ["boto3.resources", "botocore", "urllib3.connectionpool", "s3transfer"]
    for name in annoying_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    os.makedirs("../logs", exist_ok=True)
    file_handler = logging.FileHandler(f"../logs/dynabench-server-{name}.log")
    formatter = logging.Formatter("%(name)s %(asctime)s %(msg)s")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
