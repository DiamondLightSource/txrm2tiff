import typing
import logging
import numpy as np

from .abstract import AbstractTxrm
from .ref_mixin import ReferenceMixin
from .annot_mixin import AnnotatorMixin
from .save_mixin import SaveMixin
from .txrm_property import txrm_property
from ..txrm_functions.images import fallback_image_interpreter
from ..utils.image_processing import stitch_images


class Txrm5(SaveMixin, ReferenceMixin, AnnotatorMixin, AbstractTxrm):
    @txrm_property(fallback=None)
    def is_mosaic(self) -> bool:
        return bool(np.sum(self.mosaic_dims))

    @txrm_property(fallback=[])
    def image_dims(self) -> typing.List[int]:
        return [
            self.image_info["ImageWidth"][0],
            self.image_info["ImageHeight"][0],
        ]

    @txrm_property(fallback=[])
    def reference_dims(self):
        return [
            self.reference_info["ImageWidth"][0],
            self.reference_info["ImageHeight"][0],
        ]

    @txrm_property(fallback=None)
    def reference_exposure(self) -> typing.Optional[float]:
        return self.reference_info["ExpTime"][0]

    def extract_reference_image(self) -> np.ndarray:
        try:
            ref_data = self.extract_reference_data()

            if isinstance(ref_data, bytes):  # If unable to extract dtype
                ref_data = fallback_image_interpreter(
                    ref_data, np.prod(self.reference_dims), self.strict
                )

            return ref_data.reshape(self.reference_dims[::-1])
        except Exception:
            if self.strict:
                raise
            logging.error("Error occurred extracting reference image", exc_info=True)

    def get_output(
        self, load: bool = False, flip: bool = True, clear_images: bool = True
    ) -> typing.Optional[np.ndarray]:
        """
        Returns output image as ndarray with axes [idx, y, x]. If a reference has been applied, the referenced image will be returned.

        load: load the image(s) if they are not already loaded. Does not apply any reference.
        flip: flip the Y-axis of the output image(s) (how they are displayed in DX)
        clear_images: clear images and reference from the Txrm instance after returning.
        """
        images = self.get_images(load)
        if images is None:
            logging.warning("No image has been loaded, so no output can be returned.")
            return None
        images = images.copy()
        if self.is_mosaic:
            images = stitch_images(images, self.mosaic_dims)
        if clear_images:
            self.clear_images()
            self.clear_reference()

        if flip:
            return np.flip(images, axis=1)
        return images
