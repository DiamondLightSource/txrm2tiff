from __future__ import annotations
from abc import ABC
from datetime import datetime
import re
import logging
import numpy as np
from typing import TYPE_CHECKING

from ...xradia_properties.enums import XrmDataTypes
from ... import txrm_functions
from ..wrappers import txrm_property
from ...utils.exceptions import TxrmFileError
from .file import TxrmFile

if TYPE_CHECKING:
    from typing import Any, cast


datetime_regex = re.compile(
    r"(\d{2})\/(\d{2})\/(\d{2})?(\d{2}) (\d{2}):(\d{2}):(\d{2})(\.(\d{2}))?"
)
# Groups "1, 2, 4, 5, 6, 7, 9" are "mm, dd, yy, hh, mm, ss, ff"
# (months, days, year, hours, minutes, seconds, decimal of seconds)
# Note that v3 files do not include the century digits.


class TxrmWithMetadata(TxrmFile, ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @txrm_property(fallback=dict())
    def image_info(self) -> dict[str, list[Any]]:
        return txrm_functions.get_image_info_dict(
            self._get_ole_if_open(), ref=False, strict=self.strict
        )

    @txrm_property(fallback=dict())
    def reference_info(self) -> dict[str, list[Any]]:
        return txrm_functions.get_image_info_dict(
            self._get_ole_if_open(), ref=True, strict=self.strict
        )

    @txrm_property(fallback=dict())
    def position_info(self) -> dict[str, tuple[list[float], str]]:
        return txrm_functions.get_position_dict(self._get_ole_if_open())

    @txrm_property(fallback=None)
    def version(self) -> float | None:
        return txrm_functions.get_file_version(
            self._get_ole_if_open(), strict=self.strict
        )

    @txrm_property(fallback=None)
    def mosaic_dims(self) -> tuple[int, int]:
        """Returns List of mosaic dims [x columns, y rows]"""
        if self.image_info is None:
            raise TxrmFileError("Failed to get image_info")
        return (
            self.image_info["MosiacColumns"][0],
            self.image_info["MosiacRows"][0],
        )

    @txrm_property(fallback=[])
    def energies(self) -> list[float]:
        energies = self.image_info["Energy"]
        if not np.sum(energies):
            # position_info includes units
            energies = self.position_info["Energy"][0]
        if np.sum(energies):
            return energies
        raise ValueError("Could not get energies")

    @txrm_property(fallback=[])
    def exposures(self) -> list[float]:
        """Returns list of exposures"""
        if "ExpTimes" in self.image_info:
            return self.image_info["ExpTimes"]
        elif "ExpTime" in self.image_info:
            return self.image_info["ExpTime"]
        elif self.strict:
            raise KeyError("No exposure time available in ole file.")
        logging.error("No exposure time available in ole file.")
        return []

    @txrm_property(fallback=None)
    def has_reference(self) -> bool | None:
        return self.has_stream("ReferenceData/Image")

    @txrm_property(fallback=[])
    def datetimes(self) -> list[datetime]:
        dates = []
        for date_str in self.image_info["Date"]:
            m = datetime_regex.search(date_str)
            if m:  # Ignore out any random characters that aren't dates
                if m.group(3):  # Century digits
                    pattern = r"%m%d%Y%H%M%S"
                else:
                    pattern = r"%m%d%y%H%M%S"
                if m.group(9):  # Fraction of a second
                    pattern += r"%f"
                date_str = "".join([d for d in m.group(1, 2, 3, 4, 5, 6, 7, 9) if d])
                dates.append(datetime.strptime(date_str, pattern))
        return dates

    @txrm_property(fallback=None)
    def image_dtype(self) -> XrmDataTypes | None:
        return XrmDataTypes.from_number(
            cast(int, self.read_single_value_from_stream("ImageInfo/DataType")),
        )

    @txrm_property(fallback=None)
    def reference_dtype(self) -> XrmDataTypes | None:
        return XrmDataTypes.from_number(
            cast(int, self.read_single_value_from_stream("ReferenceData/DataType")),
        )

    @property
    def metadata(self) -> Any:
        """Metadata for saving"""
        return None
