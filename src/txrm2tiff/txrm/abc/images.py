from __future__ import annotations
from abc import abstractmethod, ABC
import logging
import itertools
from pathlib import Path
import numpy as np
import tifffile as tf
from olefile import isOleFile  # type: ignore[import-untyped]
from ome_types import from_xml
from tifffile.tifffile import TiffFileError
import typing

from .metadata import TxrmWithMetadata
from ..wrappers import txrm_property
from ..main import open_txrm
from ... import txrm_functions
from ...xradia_properties.enums import XrmDataTypes
from ...utils.image_processing import cast_to_dtype
from ...utils.exceptions import TxrmError, TxrmFileError
from ...utils.file_handler import manual_save
from ...utils.file_handler import file_can_be_opened
from ...utils.image_processing import dynamic_despeckle_and_average_series
from ...utils.functions import conditional_replace
from ...utils.exceptions import Txrm2TiffIOError


if typing.TYPE_CHECKING:
    from os import PathLike
    from numpy.typing import NDArray, DTypeLike

    T = typing.TypeVar("T", bound=typing.Any)


class TxrmWithImages(TxrmWithMetadata, ABC):

    def __init__(
        self,
        load_images: bool = True,
        load_reference: bool = True,
        *args: typing.Any,
        **kwargs: typing.Any,
    ):
        super().__init__(*args, **kwargs)
        self._images: NDArray[typing.Any] | None = None
        self._reference: NDArray[typing.Any] | None = None
        self.referenced: bool = False

        if load_images:
            self.load_images()
        if load_reference and self.has_reference:
            self.load_reference()

    def load_images(self) -> None:
        try:
            if not self.file_is_open:
                logging.error("Cannot read image while file is closed")
            logging.info("Getting images from the file")
            images = self.extract_images()
            if images is not None:
                self._images = images
                self.referenced = False
        except IOError:
            logging.error("Failed to load images", exc_info=True)
            if self.strict:
                raise

    def get_images(self, load: bool = True) -> NDArray[typing.Any] | None:
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
            self._reference = self._extract_single_image(
                "ReferenceData/Image",
                dtype=self.reference_dtype,
                fallback_shape=self.reference_dims,
            )
        except KeyError:
            if self.strict:
                raise
            logging.warning("No reference is available to load")
        except Exception:
            if self.strict:
                raise
            logging.error("Error occurred extracting reference image", exc_info=True)

    def get_reference(self, load: bool = True) -> NDArray[typing.Any] | None:
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

    def _extract_single_image(
        self,
        img_key: str,
        dtype: XrmDataTypes | None,
        fallback_shape: (
            NDArray[np.integer[typing.Any]] | list[int] | tuple[int, ...] | None
        ) = None,
        strict: bool = False,
    ) -> NDArray[typing.Any]:
        try:
            imgdata = None
            if not self.has_stream(img_key):
                raise KeyError("Stream '%s' does not exist" % img_key)
            if self._ole is None:
                raise TxrmFileError("OLE file has not been defined")
            img_stream_bytes = self._ole.openstream(img_key).getvalue()
            if dtype is not None:
                try:
                    imgdata = txrm_functions.get_stream_from_bytes(
                        img_stream_bytes, dtype=dtype.value
                    )
                except Exception:
                    logging.error(
                        "Image could not be extracted using expected dtype '%s'",
                        dtype,
                    )
            if (
                imgdata is None and fallback_shape is not None
            ):  # if dtype was not given or that method failed
                img_size = np.prod(fallback_shape)
                imgdata = txrm_functions.fallback_image_interpreter(
                    img_stream_bytes, img_size, self.strict
                )
                if imgdata.size > 0:
                    imgdata = imgdata.reshape(fallback_shape)  # Resize if not empty
            if imgdata is None:
                raise TxrmError(f"Failed to get image data from '{img_key}'")
            return imgdata
        except Exception:
            if strict:
                raise
            return np.asarray([])

    def extract_images(
        self,
        start: int = 1,
        end: int | None = None,
    ) -> NDArray[typing.Any]:
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
            images_taken = int(self.image_info.get("ImagesTaken", [0])[0])

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

            shape = self.shape
            if shape is None or np.prod(shape) * images_taken == 0:
                raise AttributeError("No images found")
            # Iterates through images until the number of images taken
            # lambda check has been left in in case stream is wrong
            images = (
                self._extract_single_image(
                    txrm_functions.get_image_key(i),
                    dtype=self.image_dtype,
                    fallback_shape=self.shape,
                    strict=False,
                )
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

    def get_central_image(self) -> NDArray[typing.Any]:
        """Returns central image of an odd count stack, or the first of the two central images if an even number"""
        images_taken = self.image_info["ImagesTaken"][0]
        central_img = 1 + images_taken // 2  # First image is 1
        return self.get_single_image(central_img)

    @txrm_property(fallback=0)
    def zero_angle_index(self) -> int:
        angles = self.image_info.get("Angles", [])
        if len(self.exposures) <= 1 or len(np.unique(angles)) <= 1:
            # If only a single (or no) exposure or sample theta is consitent, return 0
            return 0
        # Return index of angle closest to 0
        return int(np.array([abs(angle) for angle in angles]).argmin())

    def get_single_image(self, idx: int) -> NDArray[typing.Any]:
        """Get a single image (from memory if images are loaded). idx starts from 1."""
        images = self.get_images(load=False)
        if images is not None:
            image = images[idx - 1]
            assert isinstance(image, np.ndarray)
            return image
        else:
            return self._extract_single_image(
                txrm_functions.get_image_key(idx), self.image_dtype, strict=self.strict
            )

    @txrm_property(fallback=None)
    def shape(self) -> tuple[int, int]:
        """Shape of the stored image(s) in the order (y row, x col). This is the shape of the whole mosaic if a stitched image is stored."""
        if self._images is None:
            return (self.image_info["ImageHeight"][0], self.image_info["ImageWidth"][0])
        shape = self._images.shape
        return (shape[-2], shape[-1])

    @txrm_property(fallback=None)
    def output_shape(self) -> tuple[int, int, int]:
        """
        Returns shape that the output file will be in numpy ordering [idx, y, x]
        """
        dims = self.image_dims
        assert dims is not None
        if self.is_mosaic:
            mosaic_dims = self.mosaic_dims
            assert mosaic_dims is not None
            dims = (dims[0] * mosaic_dims[0], dims[1] * mosaic_dims[1])
            stack = 1
        else:
            stack = self.image_info["ImagesTaken"][0]
        return (stack, dims[1], dims[0])

    @txrm_property(fallback=None)
    @abstractmethod
    def is_mosaic(self) -> bool:
        """Functions overriding this function need to use the wrapper txrm_property"""
        raise NotImplementedError

    @txrm_property(fallback=None)
    @abstractmethod
    def image_dims(self) -> tuple[int, int]:
        """
        Dimensions of a single image

        Functions overriding this function need to use the wrapper txrm_property
        """
        raise NotImplementedError

    @txrm_property(fallback=None)
    @abstractmethod
    def reference_dims(self) -> tuple[int, int]:
        """
        Dimensions of the reference image

        Functions overriding this function need to use the wrapper txrm_property
        """
        raise NotImplementedError

    @txrm_property(fallback=None)
    @abstractmethod
    def reference_exposure(self) -> float | None:
        """Functions overriding this function need to use the wrapper txrm_property"""
        raise NotImplementedError

    @abstractmethod
    def extract_reference_image(self) -> NDArray[typing.Any]:
        raise NotImplementedError

    @abstractmethod
    def get_output(
        self,
        load: bool = False,
        shifts: bool = False,
        flip: bool = False,
        clear_images: bool = True,
    ) -> NDArray[typing.Any] | None:
        raise NotImplementedError

    def set_dtype(
        self,
        dtype: DTypeLike,
        allow_clipping: bool = False,
    ) -> bool:
        if self._images is None:
            logging.error("Images must be loaded before a datatype can be set.")
            return False
        self._images = cast_to_dtype(self._images, dtype, allow_clipping=allow_clipping)
        return True

    def save_images(
        self,
        filepath: str | PathLike[str] | None = None,
        datatype: DTypeLike | None = None,
        shifts: bool = False,
        flip: bool = False,
        clear_images: bool = False,
        mkdir: bool = False,
        strict: bool | None = None,
    ) -> Path | None:
        """Saves images (if available) returning True if successful."""
        if strict is None:
            strict = self.strict
        try:
            if filepath is None:
                if self.path is None:
                    raise ValueError(
                        "An output filepath must be given if an input path was not given."
                    )
                filepath = self.path.resolve().with_suffix(".ome.tiff")

            filepath = Path(filepath)
            if not self.referenced:
                logging.info("Saving without reference")

            im = self.get_output(
                load=True, shifts=shifts, flip=flip, clear_images=clear_images
            )
            if im is None:
                raise AttributeError("Cannot save image as no image has been loaded.")
            if datatype is not None:
                self.set_dtype(datatype)

            if mkdir:
                filepath.parent.mkdir(parents=True, exist_ok=True)

            manual_save(filepath, im, self.metadata)
            return filepath
        except Exception:
            logging.error("Saving failed", exc_info=not strict)
            if strict:
                raise
            return None

    def apply_reference(
        self,
        custom_reference: str | PathLike[str] | None = None,
        compensate_exposure: bool = True,
        overwrite: bool = True,
    ) -> NDArray[typing.Any] | None:
        if custom_reference is not None and file_can_be_opened(custom_reference):
            if isOleFile(str(custom_reference)):
                with open_txrm(custom_reference) as ref_txrm:
                    self.apply_reference_from_txrm(ref_txrm)
            else:
                return self._apply_reference_from_tiff(
                    custom_reference, compensate_exposure, overwrite
                )
        elif self.has_reference:
            return self.apply_internal_reference(
                overwrite=overwrite, compensate_exposure=compensate_exposure
            )
        else:
            logging.info("No reference to apply")
        return None

    def _apply_reference_from_tiff(
        self,
        custom_reference: str | PathLike[str],
        compensate_exposure: bool = True,
        overwrite: bool = True,
    ):
        try:
            with tf.TiffFile(str(custom_reference)) as tif:
                kwargs: dict[str, typing.Any] = {}
                if compensate_exposure:
                    try:
                        if not isinstance(tif.pages[0], tf.TiffPage):
                            raise Txrm2TiffIOError(
                                "TIFF structure incompatible, TIFF pages are required to read metadata"
                            )
                        pixels = from_xml(tif.pages[0].description).images[0].pixels
                        kwargs["custom_exposure"] = np.mean(
                            [
                                plane.exposure_time
                                for plane in pixels.planes
                                if plane.exposure_time is not None
                            ]
                        )
                    except Exception:
                        logging.warning(
                            "Unable to extract exposure(s) from TIFF metadata - reference will not be scaled.",
                            exc_info=True,
                        )
                arr = tif.asarray()
                return self.apply_custom_reference_from_array(
                    arr, overwrite=overwrite, **kwargs
                )
        except TiffFileError:
            logging.error(
                "Invalid custom reference supplied: file is not a TXRM or TIFF file."
            )
            return

    def apply_custom_reference_from_array(
        self,
        custom_reference: NDArray[typing.Any],
        custom_exposure: float | None = None,
        overwrite: bool = True,
    ) -> NDArray[typing.Any]:
        """Applies numpy array as a reference image to txrm image(s), returning the images as a numpy ndarray and overwrites the txrm image(s) if overwrite is True."""
        if self.referenced:
            logging.warning(
                "Applying reference to already referenced txrm image(s). If you did not mean to reference more than once, reload the images before applying the correct reference."
            )
        try:
            ref_img = self._tile_reference_if_needed(
                TxrmWithImages._flatten_reference(custom_reference)
            )
        except Exception:
            if self.strict:
                raise
            logging.error(
                "Exception occurred preparing the reference for %s",
                self.name,
                exc_info=True,
            )
        if custom_exposure is not None:
            self._compensate_ref_exposure(ref_img, custom_exposure)
        images = self.get_images()
        if images is None:
            raise TxrmError("No images found to apply reference to")
        ref = TxrmWithImages._apply_reference_to_images(images, ref_img)
        if overwrite:
            self._images = ref
            self.referenced = True
        return ref

    def apply_reference_from_txrm(
        self,
        txrm: TxrmWithImages,
        compensate_exposure: bool = True,
        overwrite: bool = True,
    ) -> NDArray[typing.Any]:
        """Applies image(s) to txrm image(s), returning the images as a numpy ndarray and overwrites the txrm image(s) if overwrite is True."""
        images = txrm.get_images(load=True)
        if images is None:
            raise TxrmError("Failed to get images to apply reference to")
        mean_exposure = float(np.mean(txrm.exposures, dtype=np.float64))
        return self.apply_custom_reference_from_array(
            images,
            custom_exposure=(mean_exposure if compensate_exposure else None),
            overwrite=overwrite,
        )

    def apply_internal_reference(
        self, compensate_exposure: bool = True, overwrite: bool = True
    ) -> NDArray[typing.Any] | None:
        """Applies reference to txrm image(s), returning the images as a numpy ndarray and overwrites the txrm image(s) if overwrite is True."""
        if self.referenced:
            logging.warning(
                "Applying reference to already referenced txrm image(s). If you did not mean to reference more than once, reload the images before applying the correct reference."
            )
        ref_img = self.get_reference(load=True)
        logging.info("Internal reference will be applied to %s", self.name)
        if ref_img is None:
            if self.strict:
                raise AttributeError("No internal reference to apply")
            logging.warning("No internal reference to apply")
            return None
        if compensate_exposure and self.reference_exposure is not None:
            ref_img = self._compensate_ref_exposure(ref_img, self.reference_exposure)
        images = self.get_images(load=True)
        if images is None:
            if self.strict:
                raise AttributeError(
                    "Cannot apply reference to images that cannot be found"
                )
            logging.warning("Cannot apply reference to images that cannot be found")
            return None
        referenced = TxrmWithImages._apply_reference_to_images(images, ref_img)
        if overwrite:
            self._images = referenced
            self.referenced = True
        return referenced

    def _tile_reference_if_needed(
        self, custom_reference: NDArray[typing.Any]
    ) -> NDArray[typing.Any]:
        """Tile the image if it is needed to match a mosaic Assumes axes [y, x]."""
        mosaic_dims = self.mosaic_dims
        assert mosaic_dims is not None
        shape = self.shape
        assert shape is not None

        needs_stitching = self.is_mosaic and custom_reference.shape == [
            round(img_dim / mos_dim, 2)
            for img_dim, mos_dim in zip(shape, mosaic_dims[::-1])
        ]  # True if it's a mosaic and the correct dims to be tiled to mosaic
        assert custom_reference.shape == shape or needs_stitching, (
            "Invalid reference shape for %s" % self.name
        )
        # Checks that reference is either the size of the image or can be stitched to that size
        if needs_stitching:
            return np.tile(
                custom_reference, mosaic_dims[::-1]
            )  # Tiles reference if needed
        return custom_reference

    def _compensate_ref_exposure(
        self, ref_image: NDArray[typing.Any], ref_exposure: float
    ) -> NDArray[typing.Any]:
        # TODO: Improve this in line with ALBA's methodolgy
        # Normalises the reference exposure
        # Assumes roughly linear response, which is a reasonable estimation
        # because exposure times are unlikely to be significantly different)
        # (if it is a tomo, it does this relative to the 0 degree image, not on a per-image basis)
        multiplier = self.exposures[self.zero_angle_index] / ref_exposure
        return ref_image.copy() * multiplier

    @staticmethod
    def _apply_reference_to_images(
        images: NDArray[typing.Any], reference: NDArray[typing.Any]
    ) -> NDArray[typing.Any]:
        floated_and_referenced = (
            np.asarray(images, dtype=np.float32) * 100.0 / reference
        )
        if (
            np.isnan(floated_and_referenced).any()
            or np.isinf(floated_and_referenced).any()
        ):
            logging.warning(
                "Potential dead pixels found. "
                "NaN was output for at least one pixel in the referenced image."
            )
            # Replace any infinite pixels (nan or inf) with 0:
            conditional_replace(floated_and_referenced, 0, lambda x: ~np.isfinite(x))
        return floated_and_referenced.astype(np.float32)

    @staticmethod
    def _flatten_reference(image: NDArray[typing.Any]) -> NDArray[typing.Any]:
        """
        Despeckle and average images if they are an image stack. Assumes axes [idx, y, x] and tile the image if needed.

        Returns numpy array with axes [y, x] (flattens axes of size 1)
        """
        if len(image.shape) == 3:
            if image.shape[0] > 1:
                try:
                    image = dynamic_despeckle_and_average_series(image, average=True)
                except ValueError as e:
                    logging.error("Failed to flatten reference: %s", str(e))
                    raise
            else:
                image = np.squeeze(image)
        return image

    @txrm_property(fallback=None)
    def has_shifts(self) -> bool:
        return bool(
            self.has_stream("Alignment/X-Shifts")
            and self.has_stream("Alignment/Y-Shifts")
            and self.shifts_applied
        )

    @txrm_property(fallback=None)
    def shifts_applied(self) -> bool:
        return bool(np.any(self.x_shifts) or np.any(self.y_shifts))

    @txrm_property(fallback=[])
    def x_shifts(self) -> list[float]:
        return typing.cast(list[float], self.read_stream("Alignment/X-Shifts"))

    @txrm_property(fallback=[])
    def y_shifts(self) -> list[float]:
        return typing.cast(list[float], self.read_stream("Alignment/Y-Shifts"))

    def apply_shifts_to_images(self, images: NDArray[T]) -> NDArray[T]:
        if not self.shifts_applied:
            # if all shifts are 0, return the original image
            return images
        logging.info("Applying shifts to images")
        num_images = len(images)
        # Trim any shifts for empty images
        x_shifts = self.x_shifts[:num_images]
        y_shifts = self.y_shifts[:num_images]

        output = np.zeros_like(images)
        for im, out, x_shift, y_shift in zip(images, output, x_shifts, y_shifts):
            # Round to nearest pixel and scale to image shape
            x_shift = int(round(x_shift)) % im.shape[1]
            y_shift = int(round(y_shift)) % im.shape[0]
            if x_shift == 0 and y_shift == 0:
                out[:] = im[:]
            else:
                out[:] = np.roll(im, (y_shift, x_shift), axis=(0, 1))

        return output
