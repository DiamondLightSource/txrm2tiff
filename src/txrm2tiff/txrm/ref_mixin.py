from os import PathLike
import typing
import logging
from numbers import Number
import numpy as np
import tifffile as tf
from olefile import isOleFile
from oxdls import OMEXML
from tifffile.tifffile import TiffFileError

from . import main
from .abstract import AbstractTxrm
from ..utils.file_handler import file_can_be_opened
from ..utils.image_processing import dynamic_despeckle_and_average_series
from ..utils.functions import conditional_replace


class ReferenceMixin:
    def apply_reference(
        self,
        custom_reference: typing.Optional[typing.Union[str, PathLike]] = None,
        compensate_exposure: bool = True,
        overwrite: bool = True,
    ) -> None:
        if custom_reference is not None and file_can_be_opened(custom_reference):
            if isOleFile(str(custom_reference)):
                with main.open_txrm(custom_reference) as ref_txrm:
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

    def _apply_reference_from_tiff(
        self,
        custom_reference: PathLike,
        compensate_exposure: bool = True,
        overwrite: bool = True,
    ):
        try:
            with tf.TiffFile(str(custom_reference)) as tif:
                kwargs = {}
                if compensate_exposure:
                    try:
                        pixels = OMEXML(xml=tif.pages[0].description).image().Pixels
                        plane_count = pixels.get_plane_count()
                        kwargs["custom_exposure"] = np.mean(
                            [
                                pixels.Plane(i).get_ExposureTime()
                                for i in range(plane_count)
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
        custom_reference: np.ndarray,
        custom_exposure: typing.Optional[Number] = None,
        overwrite: bool = True,
    ) -> np.ndarray:
        """Applies numpy array as a reference image to txrm image(s), returning the images as a numpy ndarray and overwrites the txrm image(s) if overwrite is True."""
        if self.referenced:
            logging.warning(
                "Applying reference to already referenced txrm image(s). If you did not mean to reference more than once, reload the images before applying the correct reference."
            )
        try:
            ref_img = ReferenceMixin._flatten_reference(custom_reference)
            ref_img = self._tile_reference_if_needed(custom_reference)
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
        ref = ReferenceMixin._apply_reference_to_images(self.get_images(), ref_img)
        if overwrite:
            self._images = ref
            self.referenced = True
        return ref

    def apply_reference_from_txrm(
        self,
        txrm: AbstractTxrm,
        compensate_exposure: bool = True,
        overwrite: bool = True,
    ) -> np.ndarray:
        """Applies image(s) to txrm image(s), returning the images as a numpy ndarray and overwrites the txrm image(s) if overwrite is True."""
        self.apply_custom_reference_from_array(
            txrm.get_images(load=True),
            np.mean(txrm.exposures),
            compensate_exposure,
            overwrite,
        )

    def apply_internal_reference(
        self, compensate_exposure: bool = True, overwrite: bool = True
    ) -> np.ndarray:
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
            return
        if compensate_exposure:
            ref_img = self._compensate_ref_exposure(ref_img, self.reference_exposure)
        referenced = ReferenceMixin._apply_reference_to_images(
            self.get_images(load=True), ref_img
        )
        if overwrite:
            self._images = referenced
            self.referenced = True
        return referenced

    def _tile_reference_if_needed(self, custom_reference: np.ndarray) -> np.ndarray:
        """Tile the image if it is needed to match a mosaic Assumes axes [y, x]."""
        needs_stitching = self.is_mosaic and custom_reference.shape == [
            round(img_dim / mos_dim, 2)
            for img_dim, mos_dim in zip(self.shape, self.mosaic_dims[::-1])
        ]  # True if it's a mosaic and the correct dims to be tiled to mosaic
        assert custom_reference.shape == self.shape or needs_stitching, (
            "Invalid reference shape for %s" % self.name
        )
        # Checks that reference is either the size of the image or can be stitched to that size
        if needs_stitching:
            return np.tile(
                custom_reference, self.mosaic_dims[::-1]
            )  # Tiles reference if needed
        return custom_reference

    def _compensate_ref_exposure(self, ref_image, ref_exposure):
        # TODO: Improve this in line with ALBA's methodolgy
        # Normalises the reference exposure
        # Assumes roughly linear response, which is a reasonable estimation
        # because exposure times are unlikely to be significantly different)
        # (if it is a tomo, it does this relative to the 0 degree image, not on a per-image basis)
        multiplier = self.exposures[self.zero_angle_index] / ref_exposure
        return ref_image.copy() * multiplier

    @staticmethod
    def _apply_reference_to_images(
        images: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
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
    def _flatten_reference(image: np.ndarray) -> np.ndarray:
        """
        Despeckle and average images if they are an image stack. Assumes axes [idx, y, x] and tile the image if needed.

        Returns numpy array with axes [y, x] (flattens axes of size 1)
        """
        if len(image.shape) == 3:
            if image.shape[0] > 1:
                image = dynamic_despeckle_and_average_series(image, average=True)
            else:
                image = np.squeeze(image)
        return image
