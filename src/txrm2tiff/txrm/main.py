import logging
import typing
from collections import defaultdict
from os import PathLike
import olefile

from .abstract import AbstractTxrm
from .v3 import Txrm3
from .v5 import Txrm5
from .. import txrm_functions
from ..utils import file_handler


def __invalid_txrm_file(filepath: PathLike, *args) -> None:
    raise IOError(f"Invalid txrm file '{filepath}'")


txrm_classes = defaultdict(
    lambda: __invalid_txrm_file,
    {
        3: Txrm3,
        5: Txrm5,
    },
)


def open_txrm(
    filepath: PathLike,
    load_images: bool = True,
    load_reference: bool = True,
    strict: bool = False,
) -> typing.Optional[AbstractTxrm]:
    TxrmClass = get_txrm_class(filepath)
    return TxrmClass(filepath, load_images, load_reference, strict)


def get_txrm_class(
    filepath: PathLike,
) -> typing.Optional[AbstractTxrm]:
    if not file_handler.file_can_be_opened(filepath) or not file_handler.ole_file_works(
        filepath
    ):
        return __invalid_txrm_file(filepath)
    with olefile.OleFileIO(filepath) as ole:
        m_version = int(txrm_functions.get_file_version(ole))
    return txrm_classes[m_version]
