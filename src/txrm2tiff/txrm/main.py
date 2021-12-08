import typing
from collections import defaultdict
from os import PathLike
import olefile
from io import IOBase

from .abstract import AbstractTxrm
from .v3 import Txrm3
from .v5 import Txrm5
from .. import txrm_functions
from ..utils import file_handler


def __invalid_txrm_file(file: PathLike, *args) -> None:
    raise IOError(f"Invalid txrm file '{file}'")


txrm_classes = defaultdict(
    lambda: __invalid_txrm_file,
    {
        3: Txrm3,
        5: Txrm5,
    },
)


def open_txrm(
    file: typing.Union[str, PathLike, IOBase, bytes],
    load_images: bool = True,
    load_reference: bool = True,
    strict: bool = False,
) -> typing.Optional[AbstractTxrm]:
    TxrmClass = get_txrm_class(file)
    return TxrmClass(file, load_images, load_reference, strict)


def get_txrm_class(
    file: typing.Union[str, PathLike, IOBase, bytes],
) -> typing.Optional[AbstractTxrm]:
    if isinstance(file, (str, PathLike)) and (
        not file_handler.file_can_be_opened(file)
        or not file_handler.ole_file_works(file)
    ):
        return __invalid_txrm_file(file)
    try:
        ole = olefile.OleFileIO(file)
        m_version = int(txrm_functions.get_file_version(ole))
    finally:
        if not isinstance(file, IOBase):
            # Don't close the IO as it can't be reopened
            ole.close()
    return txrm_classes[m_version]
