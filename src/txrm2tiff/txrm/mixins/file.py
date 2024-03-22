from __future__ import annotations
import logging
from io import IOBase
from pathlib import Path
from typing import TYPE_CHECKING

from olefile import OleFileIO, isOleFile

from ...xradia_properties import enums, stream_dtypes
from ... import txrm_functions
from ...utils.exceptions import TxrmError, TxrmFileError

if TYPE_CHECKING:
    from typing import (
        Callable,
        Concatenate,
        TypeVar,
        Any,
        Self,
        ParamSpec,
        ParamSpecArgs,
        ParamSpecKwargs,
    )
    from types import TracebackType
    from os import PathLike
    from numpy.typing import DTypeLike

    T = TypeVar("T")
    Param = ParamSpec("Param")
    RetType = TypeVar("RetType")


class FileMixin:

    def __init__(
        self,
        /,
        strict: bool = False,
    ):
        self.path: Path | None = None
        self.name: str | None = None
        self.strict: bool = strict
        self._ole: OleFileIO | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def uses_ole(
        fn: Callable[Concatenate[FileMixin, Param], RetType],
    ) -> Callable[Concatenate[FileMixin, Param], RetType | None]:
        def wrapped(self: FileMixin, /, *args: Any, **kwargs: Any) -> RetType | None:
            strict = kwargs.get("strict", self.strict)
            try:
                return fn(*args, **kwargs)
            except TxrmFileError:
                if strict:
                    raise TxrmFileError(f"{fn.__name__}: failed as file is not open.")
                return None

        return wrapped

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

    @uses_ole
    def has_stream(self, key: str) -> bool | None:
        exists: bool = self._get_ole_if_open().exists(key)
        return exists

    @uses_ole
    def list_streams(self) -> list[str]:
        ole = self._get_ole_if_open()
        return [
            "/".join(stream) for stream in ole.listdir(streams=True, storages=False)
        ]

    @uses_ole
    def read_stream(
        self,
        key: str,
        dtype: enums.XrmDataTypes | DTypeLike | None = None,
        strict: bool | None = None,
    ) -> list[str | float | int | bytes] | None:
        if strict is None:
            strict = self.strict
        if dtype is None:
            dtype = stream_dtypes.streams_dict.get(key)
            if dtype is None:
                logging.error("Stream does not have known dtype, one must be specified")
                if strict:
                    raise TxrmError(
                        "Stream does not have known dtype, one must be specified"
                    )
        ole = self._get_ole_if_open()
        return txrm_functions.read_stream(ole, key, dtype, strict)

    def read_single_value_from_stream(
        self, key: str, idx: int = 0, dtype: DTypeLike | None = None
    ) -> Any | None:
        val = self.read_stream(key, dtype)
        if val is None or len(val) <= idx:
            return None
        return val[idx]
