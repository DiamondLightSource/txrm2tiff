from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import numpy as np

from ..xradia_properties import XrmDataTypes as XDTypes

from . import general

if TYPE_CHECKING:
    from typing import Any, cast
    from olefile import OleFileIO  # type: ignore[import-untyped]
    from numpy.typing import NDArray


def extract_image_dtype(
    ole: OleFileIO, key_part: str, strict: bool = False, **kwargs: Any
) -> XDTypes | None:
    key = f"{key_part}/DataType"
    integer_list = cast(
        list[int], general.read_stream(ole, key, XDTypes.XRM_INT, strict)
    )
    if not integer_list:
        return None
    return XDTypes.from_number(integer_list[0])


def extract_single_image(
    ole: OleFileIO,
    image_num: int,
    numrows: int | np.integer[Any],
    numcols: int | np.integer[Any],
    strict: bool = False,
    **kwargs: Any,
) -> NDArray[Any]:
    # Read the images - They are stored in the txrm as ImageData1 ...
    # Each folder contains 100 images 1-100, 101-200
    img_key = f"ImageData{int(np.ceil(image_num / 100.0))}/Image{image_num}"
    if not ole.exists(img_key):
        if strict:
            raise KeyError("Stream '%s' does not exist" % img_key)
        return np.asarray([])
    else:
        img_stream_bytes = ole.openstream(img_key).getvalue()
        image_dtype = extract_image_dtype(ole, "ImageInfo", strict=False)
        if image_dtype is not None:
            try:
                imgdata = general.get_stream_from_bytes(
                    img_stream_bytes, dtype=image_dtype.value
                )
            except Exception:
                logging.error(
                    "Image could not be extracted using expected dtype '%s'",
                    image_dtype,
                )
                if strict:
                    raise
        else:
            try:
                img_size = numrows * numcols
                imgdata = fallback_image_interpreter(img_stream_bytes, img_size, strict)
            except Exception:
                logging.error("Exception occurred getting %s", img_key, exc_info=True)
                if strict:
                    raise
        imgdata.shape = (int(numrows), int(numcols))
        return imgdata


def get_image_key(image_number: int) -> str:
    # They are stored in the txrm as ImageData<folder number>/Image<image number>...
    # Each folder contains 100 images 1-100, 101-200
    return f"ImageData{int(np.ceil(image_number / 100.0))}/Image{image_number}"


def fallback_image_interpreter(
    stream_bytes: bytes,
    image_size: int | np.integer[Any],
    strict: bool = False,
) -> NDArray[Any]:
    stream_length = len(stream_bytes)
    if stream_length == image_size * 2:
        dtype = XDTypes.XRM_UNSIGNED_SHORT
    elif stream_length == image_size * 4:
        dtype = XDTypes.XRM_FLOAT
    else:
        logging.error(
            "Unexpected data type with %g bytes per pixel",
            stream_length / image_size,
        )
        msg = "Image is stored as unexpected type. Expecting uint16 or float32."
        if strict:
            raise TypeError(msg)
        logging.error(msg)
        return np.asarray([])
    return np.frombuffer(stream_bytes, dtype=dtype.value)
