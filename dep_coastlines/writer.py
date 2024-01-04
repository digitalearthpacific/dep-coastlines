from dataclasses import dataclass
from typing import List, Union

from xarray import DataArray

from dep_tools.namers import DepItemPath
from dep_tools.utils import write_to_blob_storage
from dep_tools.writers import XrWriterMixin, Writer


@dataclass
class DaWriter(Writer):
    def __init__(self, itempath: DepItemPath, **kwargs):
        self._itempath = itempath
        self._kwargs = kwargs

    def write(self, da: DataArray, item_id: Union[str, List]) -> Union[str, List]:
        path = self._itempath.path(item_id)
        write_to_blob_storage(
            da, path=self._itempath.path(item_id), write_args=self._kwargs
        )
        return path
