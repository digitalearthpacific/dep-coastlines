import re
from typing import Optional

import pystac
import pystac_client
from dep_tools.namers import S3ItemPath
from cloud_logger import CsvLogger

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


# These are parsers for use with typer and argo (since argo can only use
# strings), and doesn't allow missing parameters, etc.
def int_or_none(raw: str) -> Optional[int]:
    return None if raw == "None" else int(raw)


def cs_list_of_ints(raw: str) -> list[int] | int:
    return [int(s) for s in raw.split(",")] if "," in raw else int(raw)


def bool_parser(raw: str):
    return False if raw == "False" else True


def use_alternate_s3_href(modifiable: pystac_client.Modifiable) -> None:
    if isinstance(modifiable, dict):
        if modifiable["type"] == "FeatureCollection":
            new_features = list()
            for item_dict in modifiable["features"]:
                use_alternate_s3_href(item_dict)
                new_features.append(item_dict)
            modifiable["features"] = new_features
        else:
            stac_object = pystac.read_dict(modifiable)
            use_alternate_s3_href(stac_object)
            modifiable.update(stac_object.to_dict())
    else:
        for _, asset in modifiable.assets.items():
            asset_dict = asset.to_dict()
            if "alternate" in asset_dict.keys():
                asset.href = asset.to_dict()["alternate"]["s3"]["href"]
