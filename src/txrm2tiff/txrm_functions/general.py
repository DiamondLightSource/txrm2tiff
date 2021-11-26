import logging
import re
import typing
import numbers
import numpy as np
import olefile as of
import numpy.typing as npt

from .. import xradia_properties as xp

hex_pattern = re.compile(r"[^\x20-\x7e]+")


def read_stream(
    ole: of.OleFileIO,
    key: str,
    dtype: typing.Optional[typing.Union[xp.XrmDataTypes, npt.DTypeLike, None]] = None,
    strict: bool = False,
) -> typing.List[typing.Any]:
    """Reads and returns list containing stream specified by key, decoded as dtype"""
    try:
        if dtype is None:
            dtype = xp.streams_dict.get(key)
            if dtype is None:
                raise TypeError("No known data type found, one must be specified.")
        if isinstance(dtype, xp.XrmDataTypes):
            # streams_dict returns XrmDataTypes
            dtype = dtype.value
        dtype = np.dtype(dtype)  # cast to numpy dtype
        if ole.exists(key):
            if dtype == np.str_:
                return _read_text_stream_to_list(ole, key)
            return _read_number_stream_to_list(ole, key, dtype)
        raise KeyError("Stream %s does not exist in ole file" % key)
    except Exception:
        if strict:
            raise
        logging.error(
            "Error occurred reading stream '%s' as %s", key, dtype, exc_info=True
        )
        return []


def get_stream_from_bytes(stream_bytes: bytes, dtype: npt.DTypeLike) -> np.ndarray:
    """Converts olefile bytes to np.ndarray of valuess of type dtype."""
    return np.frombuffer(stream_bytes, dtype)


def _read_number_stream_to_list(
    ole: of.OleFileIO, key: str, dtype: typing.Union[npt.DTypeLike, None]
) -> typing.List[typing.Union[numbers.Number, bytes]]:
    """Reads olefile stream and returns to list of values of type dtype."""
    stream_bytes = ole.openstream(key).getvalue()
    return get_stream_from_bytes(stream_bytes, dtype).tolist()


def _read_text_stream_to_list(ole: of.OleFileIO, key: str) -> typing.List[str]:
    """
    Returns list of strings.

    The byte string is decoded then split by hex bytes.
    This is because text is stored in byte strings with seemingly random hex.

    Splitting by hex keeps real spaces while splitting entries (such as with Date).
    """
    if ole.exists(key):
        byte_str = ole.openstream(key).read()
        return [
            s for s in hex_pattern.split(byte_str.decode("ascii", errors="ignore")) if s
        ]


def get_image_info_dict(
    ole: of.OleFileIO, ref: bool = False, strict: bool = False
) -> typing.Dict[str, typing.List[typing.Any]]:
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
    ole: of.OleFileIO,
) -> typing.Dict[str, typing.Tuple[typing.Union[typing.List, str]]]:
    """
    Gets dictionary of motor posisions.

    Args:
        ole (of.OleFileIO)

    Returns:
        typing.Dict[str, typing.Tuple[typing.Union[typing.List, str]]]: all positions for each motor in the form: {motor name string: ([values], unit string)}.
    """
    tmp = {}
    for key, dtype in xp.stream_dtypes.position_info_dict.items():
        tmp[key] = read_stream(ole, key, dtype)

    num_axes = tmp["PositionInfo/TotalAxis"][0]
    positions_dict = {}
    # Motor positions are stored in list of all values for all images. It lists
    # all axes for each frame (in that order).
    for i in range(num_axes):
        positions_dict[tmp["PositionInfo/AxisNames"][i]] = (
            tmp["PositionInfo/MotorPositions"][i::num_axes],
            tmp["PositionInfo/AxisUnits"][i],
        )
    return positions_dict


def get_file_version(ole: of.OleFileIO, strict: bool = False) -> typing.Optional[float]:
    v = read_stream(ole, "Version", xp.XrmDataTypes.XRM_FLOAT, strict=strict)
    if v:
        return v[0]
    return None
