from __future__ import annotations
from collections import defaultdict
from os import PathLike
import olefile
from io import IOBase
from typing import TYPE_CHECKING

from .abstract import AbstractTxrm
from .v3 import Txrm3
from .v5 import Txrm5
from .. import txrm_functions
from ..utils import file_handler

if TYPE_CHECKING:
    from typing import Any, Never


def __invalid_txrm_file(f: Any, *args) -> Never:
    raise IOError(f"Invalid txrm file '{f}'")


txrm_classes: dict[int, AbstractTxrm] = {
    3: Txrm3,
    5: Txrm5,
}


def open_txrm(
    f: str | PathLike[Any] | IOBase | bytes,
    load_images: bool = True,
    load_reference: bool = True,
    strict: bool = False,
) -> AbstractTxrm:
    TxrmClass = get_txrm_class(f)
    return TxrmClass(f, load_images, load_reference, strict)


def get_txrm_class(
    f: str | PathLike[Any] | IOBase | bytes,
) -> AbstractTxrm | Callable:
    if isinstance(f, (str, PathLike)) and (
        not file_handler.file_can_be_opened(f) or not file_handler.ole_file_works(f)
    ):
        return __invalid_txrm_file(f)
    try:
        ole = olefile.OleFileIO(f)
        m_version = int(txrm_functions.get_file_version(ole))
    finally:
        if not isinstance(f, IOBase):
            # Don't close the IO as it can't be reopened
            ole.close()
    return txrm_classes.get(m_version, __invalid_txrm_file)
