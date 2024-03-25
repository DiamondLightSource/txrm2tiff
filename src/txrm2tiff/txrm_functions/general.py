from __future__ import annotations
import logging
import re
import numpy as np
from typing import TYPE_CHECKING

from .. import xradia_properties as xp

if TYPE_CHECKING:
    from typing import Any, Never, TypeVar, cast
    from numpy.typing import DTypeLike, NDArray
    from olefile import OleFileIO

    U = TypeVar("U", bound=Any)

hex_pattern = re.compile(r"[^\x20-\x7e]+")


def read_stream(
    ole: OleFileIO,
    key: str,
    dtype: xp.XrmDataTypes | DTypeLike | None = None,
    strict: bool = False,
) -> list[str] | list[float] | list[int] | list[bytes] | list[Never]:
    """Reads and returns list containing stream specified by key, decoded as dtype"""
    try:
        if dtype is None:
            dtype = xp.streams_dict.get(key)
            if dtype is None:
                raise TypeError("No known data type found, one must be specified.")
        if isinstance(dtype, xp.XrmDataTypes):
            # streams_dict returns XrmDataTypes
            dtype = cast(np.dtype[Any], dtype.value)
        dtype = np.dtype(dtype)  # cast to np.dtype
        if ole.exists(key):
            if np.issubdtype(dtype, np.str_):
                return _read_text_stream_to_list(ole, key)
            elif (
                np.issubdtype(dtype, np.integer)
                or np.issubdtype(dtype, np.floating)
                or np.issubdtype(dtype, np.bytes_)
            ):
                return _read_number_stream_to_list(ole, key, dtype)
            else:
                raise TypeError(f'Cannot interpret stream with type "{dtype}"')

        raise KeyError("Stream %s does not exist in ole file" % key)
    except Exception:
        if strict:
            raise
        logging.error(
            "Error occurred reading stream '%s' as %s", key, dtype, exc_info=True
        )
        return []


def get_stream_from_bytes(stream_bytes: bytes, dtype: np.dtype[U]) -> NDArray[U]:
    """Converts olefile bytes to np.ndarray of values of type dtype."""
    return np.frombuffer(stream_bytes, dtype)


def _read_number_stream_to_list(
    ole: OleFileIO, key: str, dtype: np.dtype[Any]
) -> list[Any]:
    """Reads olefile stream and returns to list of values of type dtype."""
    stream_bytes = ole.openstream(key).getvalue()
    return get_stream_from_bytes(stream_bytes, dtype).tolist()  # type: ignore[no-any-return]


def _read_text_stream_to_list(ole: OleFileIO, key: str) -> list[str]:
    """
    Returns list of strings.

    The byte string is decoded then split by hex bytes.
    This is because text is stored in byte strings with seemingly random hex.

    Splitting by hex keeps real spaces while splitting entries (such as with Date).
    """
    byte_str = ole.openstream(key).read()
    return [
        s for s in hex_pattern.split(byte_str.decode("ascii", errors="ignore")) if s
    ]


def get_image_info_dict(
    ole: OleFileIO, ref: bool = False, strict: bool = False
) -> dict[str, list[str] | list[float] | list[int] | list[bytes] | list[Never]]:
    """
    Reads a selection of useful ImageInfo streams from an XRM/TXRM file.

    Args:
        ole (OleFileIO): Opened XRM/TXRM file.
        ref (bool, optional): If True, returns "ReferenceData/ImageInfo/*". Defaults to False.

    Returns:
        Dict[str, List[Any]]: Dictionary containing the stream keys and their value(s), which are stored in a list.
    """
    image_info = {}
    if ref:
        info_dict = xp.stream_dtypes.ref_image_info_dict
    else:
        info_dict = xp.stream_dtypes.image_info_dict
    for key, dtype in info_dict.items():
        try:
            dict_key = key.split("/")[-1]  # No need for the whole stream
            image_info[dict_key] = read_stream(ole, key, dtype, True)
        except ValueError:
            if strict:
                raise
            logging.error("Invalid data type %s for %s", dtype, key)
        except KeyError:
            logging.debug("Key %s does not exist", key)
    return image_info


def get_position_dict(
    ole: OleFileIO,
) -> dict[str, tuple[list[float], str]]:
    """
    Gets dictionary of motor positions.

    Args:
        ole (OleFileIO)

    Returns:
        typing.Dict[str, typing.Tuple[typing.Union[typing.List, str]]]: all positions for each motor in the form: {motor name string: ([values], unit string)}.
    """
    tmp = {}
    for key, dtype in xp.stream_dtypes.position_info_dict.items():
        tmp[key] = read_stream(ole, key, dtype)

    num_axes = cast(int, tmp["PositionInfo/TotalAxis"][0])
    positions_dict = {}
    # Motor positions are stored in list of all values for all images. It lists
    # all axes for each frame (in that order).
    for i in range(num_axes):
        positions_dict[cast(str, tmp["PositionInfo/AxisNames"][i])] = (
            cast(list[float], tmp["PositionInfo/MotorPositions"][i::num_axes]),
            cast(str, tmp["PositionInfo/AxisUnits"][i]),
        )
    return positions_dict


def get_file_version(ole: of.OleFileIO, strict: bool = False) -> float | None:
    v = read_stream(ole, "Version", xp.XrmDataTypes.XRM_FLOAT, strict=strict)
    if v:
        return cast(float, v[0])
    return None
