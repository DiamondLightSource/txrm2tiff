import logging
import typing
import numpy as np
import olefile as of

from ..xradia_properties import XrmDataTypes as XDT

from . import general


def extract_image_dtype(
    ole: of.OleFileIO, key_part: str, strict: bool = False, **kwargs
) -> typing.Optional[XDT]:
    key = f"{key_part}/DataType"
    integer_list = general.read_stream(ole, key, XDT.XRM_INT, strict)
    if not integer_list:
        return None
    return XDT.from_number(integer_list[0], strict)


def extract_single_image(
    ole: of.OleFileIO,
    image_num: int,
    numrows: int,
    numcols: int,
    strict: bool = False,
    **kwargs,
) -> np.ndarray:
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
        imgdata.shape = (numrows, numcols)
        return imgdata


def fallback_image_interpreter(
    stream_bytes: bytes,
    image_size: int,
    strict: bool = False,
):
    stream_length = len(stream_bytes)
    if stream_length == image_size * 2:
        dtype = XDT.XRM_UNSIGNED_SHORT
    elif stream_length == image_size * 4:
        dtype = XDT.XRM_FLOAT
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
