from __future__ import annotations
from abc import ABC

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class TxrmBase(ABC):
    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.__txrm_properties: dict[str, Any] = {}

    def clear_all(self) -> None:
        for name in self.__txrm_properties.keys():
            delattr(self, f"_{name}")
