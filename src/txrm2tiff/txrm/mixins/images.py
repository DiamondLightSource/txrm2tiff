from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import itertools
import numpy as np
from typing import TYPE_CHECKING

from ...xradia_properties.enums import XrmDataTypes
from ...xradia_properties.stream_dtypes import streams_dict
from ...utils.metadata import get_ome_pixel_type
from ...utils.image_processing import cast_to_dtype
from .. import txrm_functions
from ..txrm_property import txrm_property
from ...utils.exceptions import TxrmError, TxrmFileError

from .file import FileMixin

if TYPE_CHECKING:
    from typing import Any, Self
    from os import PathLike
    from numpy.typing import DTypeLike, NDArray


class ImagesMixin(FileMixin, ABC):

    def __init__(
        self,
        /,
        strict: bool = False,
    ):
        self._images: NDArray[Any] | None = None
        self._reference: NDArray[Any] | None = None
        self.referenced: bool = False
        super().__init__(strict=strict)

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

    def get_images(self, load: bool = True) -> NDArray[Any] | None:
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

    def get_reference(self, load: bool = True) -> NDArray[Any] | None:
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

    @FileMixin.uses_ole
    def _extract_single_image(
        self, image_num: int, strict: bool = False
    ) -> NDArray[Any]:
        try:
            # Read the images - They are stored in the txrm as ImageData1 ...
            # Each folder contains 100 images 1-100, 101-200
            img_key = f"ImageData{int(np.ceil(image_num / 100.0))}/Image{image_num}"
            imgdata = None
            if not self.has_stream(img_key):
                raise KeyError("Stream '%s' does not exist" % img_key)
            img_stream_bytes = self._get_ole_if_open().openstream(img_key).getvalue()
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
            if imgdata.size > 0:
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
    ) -> NDArray[Any]:
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
    ) -> NDArray | bytes | None:
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
    def shape(self) -> list[int]:
        """Shape of the stored image(s) in the order (y row, x col). This is the shape of the whole mosaic if a stitched image is stored."""
        if self._images is None:
            return [self.image_info["ImageHeight"][0], self.image_info["ImageWidth"][0]]
        shape = self._images.shape
        if len(shape) > 2:
            shape = self._image[0].shape
        return shape

    @txrm_property(fallback=0)
    def zero_angle_index(self) -> float:
        angles = self.image_info.get("Angles", [])
        if len(self.exposures) <= 1 or len(np.unique(angles)) <= 1:
            # If only a single (or no) exposure or sample theta is consitent, return 0
            return 0
        # Return index of angle closest to 0
        return np.array([abs(angle) for angle in angles]).argmin()

    @txrm_property(fallback=[])
    def mosaic_dims(self) -> list[int]:
        """Returns List of mosaic dims [x columns, y rows]"""
        return [
            self.image_info["MosiacColumns"][0],
            self.image_info["MosiacRows"][0],
        ]

    @txrm_property(fallback=None)
    def has_reference(self) -> bool:
        return self.has_stream("ReferenceData/Image")

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
    def output_shape(self) -> list[int]:
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
    def image_dims(self) -> list[int]:
        """
        Dimensions of a single image

        Functions overriding this function need to use the wrapper txrm_property
        """
        raise NotImplementedError

    @txrm_property(fallback=[])
    @abstractmethod
    def reference_dims(self) -> list[int]:
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
        shifts: bool = False,
        flip: bool = False,
        clear_images: bool = True,
    ) -> NDArray[Any] | None:
        raise NotImplementedError

    def set_dtype(
        self, dtype, ensure_ome_compatability: bool = True, allow_clipping: bool = False
    ):
        if self._images is None:
            logging.error("Images must be loaded before a datatype can be set.")
            return False
        if ensure_ome_compatability:
            try:
                # Check this can be handled when saving
                get_ome_pixel_type(dtype)
            except TypeError:
                logging.error(
                    "Casting images to '%s' failed. Images will remain '%s'.",
                    dtype,
                    self._images[0].dtype,
                    exc_info=True,
                )
                return False

        self._images = cast_to_dtype(self._images, dtype, allow_clipping=allow_clipping)
        return True
