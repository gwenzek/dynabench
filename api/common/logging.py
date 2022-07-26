# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from datetime import datetime

import bottle


def init_logger(mode):
    logger = logging.getLogger()

    os.makedirs("../logs", exist_ok=True)
    file_handler = logging.FileHandler(f"../logs/dynabench-server-{mode}.log")
    formatter = logging.Formatter("%(asctime)s %(msg)s")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    annoying_loggers = ["boto3.resources", "s3transfer"]
    for name in annoying_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


@bottle.hook("after_request")
def logger_hook():
    request_time = datetime.now()
    logger.info(
        "{} {} {} {} {}".format(
            bottle.request.remote_addr,
            request_time,
            bottle.request.method,
            bottle.request.url,
            bottle.response.status,
        )
    )


logger = logging.getLogger("dynabench")
