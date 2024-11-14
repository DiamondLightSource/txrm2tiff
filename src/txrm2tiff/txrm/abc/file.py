from __future__ import annotations
import logging
import abc
from io import IOBase
from pathlib import Path
from typing import TYPE_CHECKING

from olefile import OleFileIO, isOleFile  # type: ignore[import-untyped]

from .base import TxrmBase
from ...xradia_properties import enums, stream_dtypes
from ... import txrm_functions
from ...utils.exceptions import TxrmError, TxrmFileError

from ..wrappers import uses_ole

if TYPE_CHECKING:
    from typing import (
        Any,
        Self,
        Literal,
        TypeAlias,
        overload,
    )
    from types import TracebackType
    from os import PathLike
    import numpy as np
    from numpy.typing import DTypeLike

    StrDTypeLike: TypeAlias = (
        Literal[enums.XrmDataTypes.XRM_STRING]
        | type[str]
        | type[np.str_]
        | np.dtype[np.str_]
    )

    IntDTypeLike: TypeAlias = (
        Literal[
            enums.XrmDataTypes.XRM_CHAR,
            enums.XrmDataTypes.XRM_UNSIGNED_CHAR,
            enums.XrmDataTypes.XRM_SHORT,
            enums.XrmDataTypes.XRM_UNSIGNED_SHORT,
            enums.XrmDataTypes.XRM_INT,
            enums.XrmDataTypes.XRM_UNSIGNED_INT,
            enums.XrmDataTypes.XRM_LONG,
            enums.XrmDataTypes.XRM_UNSIGNED_LONG,
        ]
        | type[int]
        | type[np.integer[Any]]
        | np.dtype[np.integer[Any]]
    )

    FloatDTypeLike: TypeAlias = (
        Literal[enums.XrmDataTypes.XRM_FLOAT, enums.XrmDataTypes.XRM_DOUBLE]
        | type[float]
        | type[np.floating[Any]]
        | np.dtype[np.floating[Any]]
    )

    BytesDTypeLike: TypeAlias = type[bytes] | type[np.bytes_] | np.dtype[np.bytes_]


class TxrmFile(TxrmBase, abc.ABC):

    def __init__(
        self,
        f: str | PathLike[Any] | IOBase | bytes,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.path: Path | None = None
        self.name: str | None = None
        self._ole: OleFileIO | None = None
        self.open(f)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    @abc.abstractmethod
    def strict(self):
        raise NotImplementedError

    def open(self, f: str | PathLike[Any] | IOBase | bytes | None = None) -> None:
        """Opens txrm file using OleFileIO. Runs on init but can be used to reopen if closed (only PathLike inputs can be reopened without specifying 'file')."""
        if self.file_is_open:
            logging.debug("File %s is already open", self.name)
        else:
            if f is None and self.path is not None:
                f = self.path
            else:
                if isinstance(f, (IOBase, bytes)):
                    f = f
                    self.path = None
                    if hasattr(f, "name"):
                        self.name = f.name
                    else:
                        self.name = str(f.__class__)
                elif isinstance(f, (str, PathLike)):
                    path = Path(f)
                    if path.exists() and isOleFile(path):
                        self.path = path
                        self.name = self.path.name
                        f = self.path
                    else:
                        raise TxrmFileError(f"'{f}' is not a valid TXRM file")
                else:
                    raise TypeError(f"Invalid type for argument 'f': {type(f)}")
            if f is not None:
                logging.debug("Opening %s", self.name)
                self._ole = OleFileIO(f)
            else:
                raise TxrmFileError("'%s' is not a valid xrm/txrm file" % self.name)
            if not self.file_is_open:
                raise TxrmFileError(
                    "'%s' failed to open for unknown reasons" % self.name
                )

    @uses_ole
    def close(self) -> None:
        """Closes txrm file. Can be reopened using open_file."""
        ole = self._get_ole_if_open()
        logging.debug("Closing file %s", self.name)
        ole.close()

    def _get_ole(self) -> OleFileIO:
        if self._ole is not None:
            return self._ole
        raise TxrmFileError("No file has been opened")

    def _get_ole_if_open(self) -> OleFileIO:
        ole = self._get_ole()
        if ole.fp is None or ole.fp.closed:
            raise TxrmFileError("File is not open")
        return ole

    @property
    def file_is_open(self) -> bool:
        try:
            self._get_ole_if_open()
            return True
        except IOError:
            return False

    @uses_ole(strict=True)
    def has_stream(self, key: str) -> bool:
        exists: bool = self._get_ole_if_open().exists(key)
        return exists

    @uses_ole(strict=True)
    def list_streams(self) -> list[str]:
        ole = self._get_ole_if_open()
        return [
            "/".join(stream) for stream in ole.listdir(streams=True, storages=False)
        ]

    @overload
    def read_stream(
        self,
        key: str,
        dtype: StrDTypeLike,
        strict: Literal[True] = ...,
    ) -> list[str]: ...

    @overload
    def read_stream(
        self,
        key: str,
        dtype: IntDTypeLike,
        strict: Literal[True] = ...,
    ) -> list[int]: ...

    @overload
    def read_stream(
        self,
        key: str,
        dtype: FloatDTypeLike,
        strict: Literal[True] = ...,
    ) -> list[float]: ...

    @overload
    def read_stream(
        self,
        key: str,
        dtype: BytesDTypeLike,
        strict: Literal[True] = ...,
    ) -> list[bytes]: ...

    @overload
    def read_stream(
        self,
        key: str,
        dtype: enums.XrmDataTypes | DTypeLike | None = ...,
        strict: Literal[True] = ...,
    ) -> list[str] | list[float] | list[int] | list[bytes]: ...

    @overload
    def read_stream(
        self,
        key: str,
        dtype: StrDTypeLike,
        strict: bool | None = ...,
    ) -> list[str] | None: ...

    @overload
    def read_stream(
        self,
        key: str,
        dtype: IntDTypeLike,
        strict: bool | None = ...,
    ) -> list[int] | None: ...

    @overload
    def read_stream(
        self,
        key: str,
        dtype: FloatDTypeLike,
        strict: bool | None = ...,
    ) -> list[float] | None: ...

    @overload
    def read_stream(
        self,
        key: str,
        dtype: BytesDTypeLike,
        strict: bool | None = ...,
    ) -> list[bytes] | None: ...

    @overload
    def read_stream(
        self,
        key: str,
        dtype: enums.XrmDataTypes | DTypeLike | None = ...,
        strict: bool | None = ...,
    ) -> list[str] | list[float] | list[int] | list[bytes] | None: ...

    def read_stream(
        self,
        key: str,
        dtype: enums.XrmDataTypes | DTypeLike | None = None,
        strict: bool | None = None,
    ) -> list[str] | list[float] | list[int] | list[bytes] | None:

        @uses_ole
        def fn(
            self: TxrmFile,
            key: str,
            dtype: enums.XrmDataTypes | DTypeLike | None = None,
            strict: bool | None = None,
        ) -> list[str] | list[float] | list[int] | list[bytes]:
            if dtype is None:
                dtype = stream_dtypes.streams_dict.get(key)
                if dtype is None:
                    raise TxrmError(
                        "Stream does not have known dtype, one must be specified"
                    )
            ole = self._get_ole_if_open()
            assert isinstance(ole, OleFileIO)
            return txrm_functions.read_stream(ole, key, dtype, True)

        return fn(self, key, dtype, strict)

    @overload
    def read_single_value_from_stream(
        self,
        key: str,
        idx: int = ...,
        dtype: enums.XrmDataTypes | DTypeLike | None = ...,
        strict: Literal[True] = ...,
    ) -> str | float | int | bytes: ...

    @overload
    def read_single_value_from_stream(
        self,
        key: str,
        idx: int = ...,
        dtype: enums.XrmDataTypes | DTypeLike | None = ...,
        strict: bool | None = ...,
    ) -> str | float | int | bytes | None: ...

    def read_single_value_from_stream(
        self,
        key: str,
        idx: int = 0,
        dtype: enums.XrmDataTypes | DTypeLike | None = None,
        strict: bool | None = None,
    ) -> str | float | int | bytes | None:
        val = self.read_stream(key, dtype, strict=strict)
        if val is None or len(val) <= idx:
            return None
        return val[idx]
