import logging
import tifffile as tf
import numpy as np
from numpy.typing import DTypeLike
from os import access, R_OK, remove, PathLike
from pathlib import Path
from typing import Optional, Union, List
from olefile import OleFileIO, isOleFile
from oxdls import OMEXML

from .metadata import dtype_dict
from .image_processing import cast_to_dtype
from ..txrm_functions import read_stream
from ..xradia_properties import XrmDataTypes as XDT
from ..info import __version__


def file_can_be_opened(path: Union[str, PathLike]) -> bool:
    if access(str(path), R_OK):
        return True
    logging.error("File %s cannot be opened", path)
    return False


def ole_file_works(path: Union[str, PathLike]) -> bool:
    path = Path(path)
    if path.is_file() and ((path.suffix == ".txrm") or (path.suffix == ".xrm")):
        if isOleFile(str(path)):
            with OleFileIO(str(path)) as ole_file:
                number_frames_taken = read_stream(
                    ole_file, "ImageInfo/ImagesTaken", XDT.XRM_INT
                )[0]
                expected_number_frames = read_stream(
                    ole_file, "ImageInfo/NoOfImages", XDT.XRM_INT
                )[0]
                # Returns true even if all frames aren't written, throwing warning.
                if number_frames_taken != expected_number_frames:
                    logging.warning(
                        "%s is an incomplete %s file: only %i out of %i frames have been written",
                        path.name,
                        path.suffix,
                        number_frames_taken,
                        expected_number_frames,
                    )
                # Check for reference frame:
                if not ole_file.exists("ReferenceData/Image"):
                    logging.warning("No reference data found in file %s", path)
                return True
        else:
            logging.warning("Could not read ole file %s", path)
    else:
        logging.warning("%s not .txrm or .xrm", path)
    return False


def manual_save(
    filepath: Union[str, PathLike],
    image: Union[np.ndarray, List[np.ndarray]],
    data_type: Optional[DTypeLike] = None,
    metadata: Optional[OMEXML] = None,
):
    filepath = Path(filepath)
    image = np.asarray(image)
    if data_type is not None:
        image = cast_to_dtype(image, data_type)
    else:
        logging.info("No data type specified. Saving with default data type.")
    if metadata is not None:
        meta_img = metadata.image()
        meta_img.Pixels.set_PixelType(dtype_dict[image.dtype.name])
        meta_img.set_Name(filepath.name)
        metadata = metadata.to_xml().encode()

    num_frames = len(image)
    bigtiff = (
        image.size * image.itemsize >= np.iinfo(np.uint32).max
    )  # Check if data bigger than 4GB TIFF limit

    logging.info("Saving image as %s with %i frames", filepath.name, num_frames)

    if filepath.exists():
        logging.warning("Overwriting existing file %s", filepath)

    with tf.TiffWriter(str(filepath), bigtiff=bigtiff, ome=False) as tif:
        tif.write(
            image,
            photometric="MINISBLACK",
            description=metadata,
            metadata={"axes": "ZYX"},
            software=f"txrm2tiff {__version__}",
        )


def manual_annotation_save(
    filepath: Union[str, PathLike], image: Union[np.ndarray, List[np.ndarray]]
):
    filepath = Path(filepath)
    num_frames = len(image)
    logging.info(
        "Saving annotated image as %s with %i frames", filepath.name, num_frames
    )
    bigtiff = (
        image.size * image.itemsize >= np.iinfo(np.uint32).max
    )  # Check if data bigger than 4GB TIFF limit

    with tf.TiffWriter(str(filepath), bigtiff=bigtiff, ome=False) as tif:
        tif.write(
            image,
            photometric="RGB",
            metadata={"axes": "ZYXC"},
            software=f"txrm2tiff {__version__}",
        )
