from __future__ import annotations
from os import PathLike
import olefile  # type: ignore[import-untyped]
from io import IOBase
from typing import TYPE_CHECKING

from .v3 import Txrm3
from .v5 import Txrm5
from .. import txrm_functions
from ..utils import file_handler

if TYPE_CHECKING:
    from typing import Any


txrm_classes: dict[int, type[Txrm3 | Txrm5]] = {
    3: Txrm3,
    5: Txrm5,
}


def open_txrm(
    f: str | PathLike[Any] | IOBase | bytes,
    load_images: bool = True,
    load_reference: bool = True,
    strict: bool = False,
) -> Txrm3 | Txrm5:
    TxrmClass = get_txrm_class(f)
    return TxrmClass(f, load_images, load_reference, strict)


def get_txrm_class(
    f: str | PathLike[Any] | IOBase | bytes,
) -> type[Txrm3 | Txrm5]:
    if isinstance(f, (IOBase | bytes)) or (
        file_handler.file_can_be_opened(f) and file_handler.ole_file_works(f)
    ):
        try:
            ole = olefile.OleFileIO(f)
            version = txrm_functions.get_file_version(ole)
            if version is None:
                raise ValueError("Unable to determine file version")
            m_version = int(version)
        finally:
            if not isinstance(f, IOBase):
                # Don't close the IO as it can't be reopened
                ole.close()
        if m_version in txrm_classes:
            return txrm_classes[m_version]

    raise IOError("Invalid txrm file'{f!r}'")
