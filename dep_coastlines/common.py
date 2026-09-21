"""Functions used in more than one place."""

import re

from cloud_logger import CsvLogger
from dep_tools.namers import S3ItemPath

import dep_coastlines.config as config


def coastlineItemPath(dataset_id: str, version: str, time: str) -> S3ItemPath:
    """Get the :class:`ItemPath` used to name coastlines data.

    Args:
        dataset_id: The dataset id. Typically a subfolder definition, for
            instance "coastlines/interim/mosaic
        version: The version.
        time: The time, e.g. "1999/2024"

    Returns: An :class:`ItemPath` ready to name output files.
    """
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
    dataset_id: str,
    delete_existing_log: bool = False,
) -> CsvLogger:
    """Get a logger.

    This is a wrapper around :func:`CsvLogger.__init__`.

    Args:
        itempath: The ItemPath used to define the log path.
        dataset_id: The dataset id.
        delete_existing_log: Whether to delete the existing log.

    Returns:
        The logger.

    """
    return CsvLogger(
        name=dataset_id,
        path=f"{itempath.bucket}/{itempath.log_path()}",
        overwrite=delete_existing_log,
        header="time|index|status|paths|comment\n",
    )
