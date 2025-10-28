import re

from cloud_logger import CsvLogger
from dep_tools.namers import S3ItemPath

import dep_coastlines.config as config


def coastlineItemPath(dataset_id: str, version: str, time: str) -> S3ItemPath:
    namer = S3ItemPath(
        bucket=config.BUCKET,
        sensor="ls",
        dataset_id=dataset_id,
        version=version,
        time=time.replace("/", "_"),
        zero_pad_numbers=True,
    )
    namer.item_prefix = re.sub("_interim|_raw|_processed", "", namer.item_prefix)
    return namer


def coastlineLogger(
    itempath: S3ItemPath,
    dataset_id,
    delete_existing_log=False,
) -> CsvLogger:
    return CsvLogger(
        name=dataset_id,
        path=f"{itempath.bucket}/{itempath.log_path()}",
        overwrite=delete_existing_log,
        header="time|index|status|paths|comment\n",
    )
