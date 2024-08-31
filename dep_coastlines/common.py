import re
from typing import Optional

from dep_tools.namers import S3ItemPath
from cloud_logger import CsvLogger, S3Handler

BUCKET = "dep-public-staging"
LOCAL_AWS_PROFILE = "dep-staging-admin"


def coastlineItemPath(dataset_id: str, version: str, time: str) -> S3ItemPath:
    namer = S3ItemPath(
        bucket=BUCKET,
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
    setup_auth: bool = False,
) -> CsvLogger:
    if setup_auth:
        from aiobotocore.session import AioSession

        handler_kwargs = dict(session=AioSession(profile=LOCAL_AWS_PROFILE))
    else:
        handler_kwargs = dict()

    return CsvLogger(
        name=dataset_id,
        path=f"{itempath.bucket}/{itempath.log_path()}",
        overwrite=delete_existing_log,
        header="time|index|status|paths|comment\n",
        cloud_handler=S3Handler,
        **handler_kwargs,
    )


def int_or_none(raw: str) -> Optional[int]:
    return None if raw == "None" else int(raw)


def cs_list_of_ints(raw: str) -> list[int] | int:
    return [int(s) for s in raw.split(",")] if "," in raw else int(raw)


def bool_parser(raw: str):
    return False if raw == "False" else True
