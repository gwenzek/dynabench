# Copyright (c) Facebook, Inc. and its affiliates.

"""
Add endpoint name to models table and
remove upload_timestamp column which is now redundant
"""

from yoyo import step


__depends__ = {"20210425_02_AtJot-remove-overall-perf"}

steps = [
    step(
        """
        ALTER TABLE models ADD COLUMN endpoint_name TEXT
        """,
        """
        ALTER TABLE models DROP endpoint_name
        """,
    ),
    step(
        """
        UPDATE models SET endpoint_name=CONCAT("ts", upload_timestamp, "-", name)
        where deployment_status="deployed"
        """,
        """
        UPDATE models SET endpoint_name=NULL where deployment_status="deployed"
        """,
    ),
    step(
        """
        ALTER TABLE models DROP upload_timestamp
        """,
        """
        ALTER TABLE models ADD COLUMN upload_timestamp BIGINT
        """,
    ),
]
