from __future__ import annotations
import logging
import tifffile as tf
import numpy as np
from os import access, R_OK
from pathlib import Path
from typing import Union, List
from olefile import OleFileIO, isOleFile  # type: ignore[import-untyped]

from ..txrm_functions import read_stream
from ..xradia_properties import XrmDataTypes as XDTypes
from .metadata import handle_tiff_resolution, get_ome_pixel_type
from ..info import __version__
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from ome_types import OME
    from numpy.typing import NDArray
    from os import PathLike


def file_can_be_opened(path: str | PathLike[str]) -> bool:
    if access(str(path), R_OK):
        return True
    logging.error("File %s cannot be opened", path)
    return False


def ole_file_works(path: str | PathLike[str]) -> bool:
    path = Path(path)
    if path.is_file() and ((path.suffix == ".txrm") or (path.suffix == ".xrm")):
        if isOleFile(str(path)):
            with OleFileIO(str(path)) as ole_file:
                number_frames_taken = read_stream(
                    ole_file, "ImageInfo/ImagesTaken", XDTypes.XRM_INT
                )[0]
                expected_number_frames = read_stream(
                    ole_file, "ImageInfo/NoOfImages", XDTypes.XRM_INT
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
    filepath: Union[str, PathLike[str]],
    image: Union[NDArray[Any], List[NDArray[Any]]],
    metadata: OME | None = None,
) -> None:
    metadata_string: str | None

    filepath = Path(filepath)
    image = np.asarray(image)
    tiff_kwargs: dict[str, Any] = {}
    if metadata is None:
        metadata_string = None
    else:
        meta_img = metadata.images[0]
        meta_img.pixels.type = get_ome_pixel_type(image.dtype)
        meta_img.name = filepath.name
        resolution_unit = (
            tf.RESUNIT.CENTIMETER
        )  # Must use CENTIMETER for maximum compatibility
        try:
            resolution = handle_tiff_resolution(metadata, resolution_unit)
            if tf.__version__ >= "2022.7.28":  # type: ignore[attr-defined]
                # 2022.7.28: Deprecate third resolution argument on write (use resolutionunit)
                tiff_kwargs["resolution"] = resolution
                tiff_kwargs["resolutionunit"] = resolution_unit
            else:
                tiff_kwargs["resolution"] = (*resolution, resolution_unit)
        except Exception:
            logging.warning(
                "Failed to include resolution info in tiff tags", exc_info=True
            )
        metadata_string = metadata.to_xml()

    num_frames = len(image)
    bigtiff = (
        image.size * image.itemsize >= np.iinfo(np.uint32).max
    )  # Check if data bigger than 4GB TIFF limit

    logging.info("Saving image as %s with %i frames", filepath.name, num_frames)

    if filepath.exists():
        logging.warning("Overwriting existing file %s", filepath)

    with tf.TiffWriter(str(filepath), bigtiff=bigtiff, ome=False, imagej=False) as tif:
        tif.write(
            image,
            photometric="MINISBLACK",
            description=metadata_string,
            metadata={"axes": "ZYX"},
            software=f"txrm2tiff {__version__}",
            **tiff_kwargs,
        )


def manual_annotation_save(filepath: str | PathLike[str], image: NDArray[Any]) -> None:
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
