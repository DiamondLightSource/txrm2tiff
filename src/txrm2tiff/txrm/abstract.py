from abc import ABC, abstractmethod
from datetime import datetime
import re
import logging
import typing
import itertools
import numpy as np
from io import IOBase
from os import PathLike
from numpy.typing import DTypeLike
from olefile import OleFileIO, isOleFile
from pathlib import Path

from ..xradia_properties.enums import XrmDataTypes
from ..xradia_properties.stream_dtypes import streams_dict
from .. import txrm_functions
from .txrm_property import txrm_property


datetime_regex = re.compile(
    r"(\d{2})\/(\d{2})\/(\d{2})?(\d{2}) (\d{2}):(\d{2}):(\d{2})(\.(\d{2}))?"
)
# Groups "1, 2, 4, 5, 6, 7, 9" are "mm, dd, yy, hh, mm, ss, ff"
# (months, days, year, hours, minutes, seconds, decimal of seconds)
# Note that v3 files do not include the century digits.


class AbstractTxrm(ABC):
    def __init__(
        self,
        file: typing.Union[str, PathLike, IOBase, bytes],
        load_images: bool = True,
        load_reference: bool = True,
        strict: bool = False,
    ):
        """Abstract class for wrapping TXRM/XRM files

        Args:
            file (str | PathLike | IOBase | bytes): Path to valid txrm file, a file-like object, or the bytes from an opened file.
            load_images (bool, optional): Load images to memory on init. Defaults to True.
            load_reference (bool, optional): Load reference images (if available) to memory on init. Defaults to True.
            strict (bool, optional): If True, all calls will be treated as strict (raising, not logging, errors). Defaults to False.
        """
        self.ole = None
        self.strict = strict
        self._images = None
        self._reference = None
        self.referenced = False
        self.annotated_image = None
        self.path = None
        self.name = None

        self.open(file)

        if load_images:
            self.load_images()
        if load_reference and self.has_reference:
            self.load_reference()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def open(
        self, file: typing.Optional[typing.Union[str, PathLike, IOBase, bytes]] = None
    ) -> None:
        """Opens txrm file using OleFileIO. Runs on init but can be used to reopen if closed (only PathLike inputs can be reopened without specifying 'file')."""
        if self.file_is_open:
            logging.debug("File %s is already open", self.name)
        else:
            f = None
            if file is None and self.path is not None:
                f = self.path
            else:
                if isinstance(file, (IOBase, bytes)):
                    f = file
                    self.path = None
                    if hasattr(file, "name"):
                        self.name = file.name
                    else:
                        self.name = f"{file.__class__}"
                elif isinstance(file, (str, PathLike)):
                    path = Path(file)
                    if path.exists() and isOleFile(path):
                        self.path = Path(file)
                        self.name = self.path.name
                        f = self.path
                    else:
                        raise IOError("Path is to invalid file")
                else:
                    raise TypeError("Invalid type for argument file")
            if f is not None:
                logging.debug("Opening file %s", self.name)
                self.ole = OleFileIO(f)
            else:
                raise IOError("'%s' is not a valid xrm/txrm file" % self.name)
            if self.ole.fp is None:
                raise IOError("'%s' failed to open for unknown reasons" % self.name)

    def close(self) -> None:
        """Closes txrm file. Can be reoped using open_file."""
        if not self.file_is_open:
            logging.info("File %s is already closed", self.name)
            return
        logging.debug("Closing file %s", self.name)
        self.ole.close()

    def clear_all(self):
        for name in self.properties.keys():
            delattr(self, f"_{name}")

    @property
    def file_is_open(self) -> bool:
        return (
            self.ole is not None and self.ole.fp is not None and not self.ole.fp.closed
        )

    def has_stream(self, key: str) -> typing.Optional[bool]:
        if not self.file_is_open:
            logging.error("Cannot check stream from closed file")
            return None
        return self.ole.exists(key)

    def list_streams(self) -> typing.List[typing.Optional[str]]:
        if not self.file_is_open:
            logging.error("Cannot list streams when file is closed")
            return []
        return [
            "/".join(stream)
            for stream in self.ole.listdir(streams=True, storages=False)
        ]

    def read_stream(
        self,
        key: str,
        dtype: typing.Optional[DTypeLike] = None,
        strict: typing.Optional[bool] = None,
    ) -> typing.List[typing.Any]:
        if not self.file_is_open:
            logging.error("Cannot get stream from closed file")
            if self.strict:
                raise IOError("Cannot get stream from closed file")
            return None
        if strict is None:
            strict = self.strict
        if dtype is None:
            dtype = streams_dict.get(key)
            if dtype is None:
                logging.error("Stream does not have known dtype, one must be specified")
                if self.strict:
                    raise IOError(
                        "Stream does not have known dtype, one must be specified"
                    )
        return txrm_functions.read_stream(self.ole, key, dtype, strict)

    def read_single_value_from_stream(
        self, key: str, idx: int = 0, dtype: typing.Optional[DTypeLike] = None
    ) -> typing.Optional[typing.Any]:
        val = self.read_stream(key, dtype)
        if val is None or len(val) <= idx:
            return None
        return val[idx]

    def load_images(self) -> None:
        try:
            if not self.file_is_open:
                logging.error("Cannot read image while file is closed")
            logging.info("Getting images from the file")
            self._images = self.extract_images()
            self.referenced = False
        except IOError:
            logging.error("Failed to load images", exc_info=True)
            if self.strict:
                raise

    def get_images(self, load: bool = True) -> typing.Optional[np.ndarray]:
        """Get images from file (numpy ndarray with shape [idx, y, x]).

        Args:
            load (bool, optional): try to load image if not already loaded. Defaults to True.

        Returns:
            Images as 3D numpy array with shape [idx, y, x], or None if not loaded.
        """
        if self._images is None and load:
            self.load_images()
        return self._images

    def clear_images(self) -> None:
        self._images = None

    def clear_reference(self) -> None:
        self._reference = None

    def load_reference(self) -> None:
        try:
            self._reference = self.extract_reference_image()
        except KeyError:
            if self.strict:
                raise
            logging.warning("No reference is available to load")
        except Exception:
            if self.strict:
                raise
            logging.error("Error occurred extracting reference image", exc_info=True)


    def get_reference(self, load: bool = True) -> typing.Optional[np.ndarray]:
        """Get images from file (numpy ndarray with shape [idx, y, x]).

        Args:
            load (bool, optional): Try to load image if not already loaded. Defaults to True.
            rescaled_by_exp (bool, optional): Rescale reference by relative exposure ()

        Returns:
            Images as 2D numpy array with shape [y, x], or None if not loaded.
        """
        if self._reference is None and load:
            self.load_reference()
        return self._reference

    def _extract_single_image(self, image_num: int, strict: bool = False) -> np.ndarray:
        try:
            # Read the images - They are stored in the txrm as ImageData1 ...
            # Each folder contains 100 images 1-100, 101-200
            img_key = f"ImageData{int(np.ceil(image_num / 100.0))}/Image{image_num}"
            imgdata = None
            if not self.has_stream(img_key):
                raise KeyError("Stream '%s' does not exist" % img_key)
            img_stream_bytes = self.ole.openstream(img_key).getvalue()
            if self.image_dtype is not None:
                try:
                    imgdata = txrm_functions.get_stream_from_bytes(
                        img_stream_bytes, dtype=self.image_dtype.value
                    )
                except Exception:
                    logging.error(
                        "Image could not be extracted using expected dtype '%s'",
                        self.image_dtype,
                    )
            if imgdata is None:  # if dtype was not given or that method failed
                img_size = np.prod(self.shape)
                imgdata = txrm_functions.fallback_image_interpreter(
                    img_stream_bytes, img_size, self.strict
                )
            if imgdata.size:
                imgdata.shape = self.shape  # Resize if not empty
            return imgdata
        except Exception:
            if strict:
                raise
            return np.asarray([])

    def extract_images(
        self,
        start: int = 1,
        end: int = None,
    ) -> np.ndarray:
        """
        Extract only specified range of images using the arguments start and num_images

        Args:
            start (int, optional): First image to extract (from 1). Defaults to 1.
            end (int, optional): Last image to extract (end = images taken if end > images taken). Defaults to extracting all images taken.
            strict (bool, optional): Log exceptions if False, raise errors if True. Defaults to False.

        Raises:
            AttributeError: No images found

        Returns:
            numpy.ndarray: Array of images with shape [idx, y, x]
        """
        try:
            images_taken = self.image_info.get("ImagesTaken", [0])[0]

            if start < 1 or start > images_taken:
                raise ValueError("Cannot extract Image%d as it does not exist" % start)

            remaining_imgs = images_taken - start + 1

            if end is None:
                # Defaults to all extracting all images
                end = images_taken
            elif end > images_taken:
                logging.warning(
                    "Cannot return up to image %d as only %d images exist, returning images %d to %d",
                    end,
                    images_taken,
                    start,
                    remaining_imgs,
                )
                end = remaining_imgs

            if np.prod(self.shape) * images_taken == 0:
                raise AttributeError("No images found")
            # Iterates through images until the number of images taken
            # lambda check has been left in in case stream is wrong
            images = (
                self._extract_single_image(i, strict=False)
                for i in range(start, end + 1)
            )
            return np.asarray(
                tuple(itertools.takewhile(lambda image: image.size > 1, images))
            )
        except Exception:
            if self.strict:
                raise
            logging.error("Error occurred extracting images", exc_info=True)
            return np.asarray([])

    def get_central_image(self) -> np.ndarray:
        """Returns central image of an odd count stack, or the first of the two central images if an even number"""
        images_taken = self.image_info["ImagesTaken"][0]
        central_img = 1 + images_taken // 2  # First image is 1
        return self.get_single_image(central_img)

    def extract_reference_data(
        self,
    ) -> typing.Optional[typing.Union[np.ndarray, bytes]]:
        """Returns 2D numpy array of reference image in row-column order"""
        # Read the reference image.
        # Reference info is stored under 'ReferenceData/...'
        if not self.has_reference:
            raise KeyError("ReferenceData/Image does not exist")
        ref_stream_bytes = self.ole.openstream("ReferenceData/Image").getvalue()
        if not self.reference_dtype:
            logging.error(
                "Image could not be extracted using expected dtype '%s', returning bytes",
                self.reference_dtype,
            )
            return ref_stream_bytes
        return txrm_functions.get_stream_from_bytes(
            ref_stream_bytes, dtype=self.reference_dtype.value
        )

    @txrm_property(fallback=[])
    def shape(self) -> typing.List[int]:
        """Shape of the stored image(s) in the order (y row, x col). This is the shape of the whole mosaic if a stitched image is stored."""
        if self._images is None:
            return [self.image_info["ImageHeight"][0], self.image_info["ImageWidth"][0]]
        shape = self._images.shape
        if len(shape) > 2:
            shape = self._image[0].shape
        return shape

    @txrm_property(fallback=dict())
    def image_info(self):
        return txrm_functions.get_image_info_dict(
            self.ole, ref=False, strict=self.strict
        )

    @txrm_property(fallback=dict())
    def reference_info(self):
        return txrm_functions.get_image_info_dict(
            self.ole, ref=True, strict=self.strict
        )

    @txrm_property(fallback=dict())
    def position_info(self):
        return txrm_functions.get_position_dict(self.ole)

    @txrm_property(fallback=None)
    def version(self):
        return txrm_functions.get_file_version(self.ole, strict=self.strict)

    @txrm_property(fallback=0)
    def zero_angle_index(self) -> float:
        angles = self.image_info.get("Angles", [])
        if len(self.exposures) <= 1 or len(np.unique(angles)) <= 1:
            # If only a single (or no) exposure or sample theta is consitent, return 0
            return 0
        # Return index of angle closest to 0
        return np.array([abs(angle) for angle in angles]).argmin()

    @txrm_property(fallback=[])
    def mosaic_dims(self) -> typing.List[int]:
        """Returns List of mosaic dims [x columns, y rows]"""
        return [
            self.image_info["MosiacColumns"][0],
            self.image_info["MosiacRows"][0],
        ]

    @txrm_property(fallback=[])
    def energies(self) -> typing.List[float]:
        energies = self.image_info["Energy"]
        if not np.sum(energies):
            energies = self.position_info["Energy"]
        if np.sum(energies):
            return energies
        raise ValueError("Could not get energies")

    @txrm_property(fallback=[])
    def exposures(self) -> typing.List[float]:
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
    def has_reference(self) -> bool:
        return self.has_stream("ReferenceData/Image")

    @txrm_property(fallback=[])
    def datetimes(self) -> typing.List[datetime]:
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

    def get_single_image(self, idx: int) -> np.ndarray:
        """Get a single image (from memory if images are loaded). idx starts from 1."""
        images = self.get_images(load=False)
        if images is not None:
            return images[idx - 1]
        else:
            return self._extract_single_image(idx, strict=self.strict)

    @txrm_property(fallback=None)
    def image_dtype(self) -> XrmDataTypes:
        return XrmDataTypes.from_number(
            self.read_single_value_from_stream("ImageInfo/DataType")
        )

    @txrm_property(fallback=None)
    def reference_dtype(self) -> XrmDataTypes:
        return XrmDataTypes.from_number(
            self.read_single_value_from_stream("ReferenceData/DataType")
        )

    @txrm_property(fallback=[])
    def output_shape(self) -> typing.List[int]:
        """
        Returns shape that the output file will be in numpy ordering [idx, y, x]

        Functions overriding this function need to use the wrapper txrm_property
        """
        dims = self.image_dims
        if self.is_mosaic:
            dims = [dim * m_dim for dim, m_dim in zip(dims, self.mosaic_dims)]
            stack = 1
        else:
            stack = self.image_info["ImagesTaken"][0]
        return [stack, *dims[::-1]]

    @txrm_property(fallback=None)
    @abstractmethod
    def is_mosaic(self) -> bool:
        """Functions overriding this function need to use the wrapper txrm_property"""
        raise NotImplementedError

    @txrm_property(fallback=[])
    @abstractmethod
    def image_dims(self) -> typing.List[int]:
        """
        Dimensions of a single image

        Functions overriding this function need to use the wrapper txrm_property
        """
        raise NotImplementedError

    @txrm_property(fallback=[])
    @abstractmethod
    def reference_dims(self) -> typing.List[int]:
        """
        Dimensions of the reference image

        Functions overriding this function need to use the wrapper txrm_property
        """
        raise NotImplementedError

    @txrm_property(fallback=None)
    @abstractmethod
    def reference_exposure(self) -> float:
        """Functions overriding this function need to use the wrapper txrm_property"""
        raise NotImplementedError

    @abstractmethod
    def extract_reference_image(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_output(
        self,
        load: bool = False,
        shifts: bool = True,
        flip: bool = False,
        clear_images: bool = True,
    ) -> typing.Optional[np.ndarray]:
        raise NotImplementedError
