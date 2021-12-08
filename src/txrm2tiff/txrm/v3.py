import typing
import logging
import numpy as np

from .abstract import AbstractTxrm
from .ref_mixin import ReferenceMixin
from .save_mixin import SaveMixin
from .shifts_mixin import ShiftsMixin
from .txrm_property import txrm_property
from ..txrm_functions.images import fallback_image_interpreter
from ..utils.functions import convert_to_int


class Txrm3(ShiftsMixin, SaveMixin, ReferenceMixin, AbstractTxrm):
    @txrm_property(fallback=None)
    def is_mosaic(self) -> bool:
        return self.image_info.get("MosiacMode", [0])[0] == 1

    @txrm_property(fallback=[])
    def image_dims(self) -> typing.List[int]:
        img_dims = [
            self.image_info["ImageWidth"][0],
            self.image_info["ImageHeight"][0],
        ]
        if self.is_mosaic:
            img_dims = self._mosaic_to_single_image_dims(img_dims)
        return img_dims

    @txrm_property(fallback=[])
    def reference_dims(self) -> typing.List[int]:
        return (
            self.image_dims
        )  # Same for both (reference width/height isn't stored separately for v3)

    def _mosaic_to_single_image_dims(self, dims) -> typing.List[int]:
        return [
            convert_to_int(dim / mos_dim)
            for dim, mos_dim in zip(dims, self.mosaic_dims)
        ]

    @txrm_property(fallback=None)
    def reference_exposure(self) -> typing.Optional[float]:
        return self.read_single_value_from_stream("ReferenceData/ExpTime")

    def extract_reference_image(self) -> np.ndarray:
        ref_data = self.extract_reference_data()

        if isinstance(ref_data, bytes):  # If unable to extract dtype
            img_size = np.prod(self.reference_dims)
            ref_data = fallback_image_interpreter(
                ref_data, int(img_size), self.strict
            )

        ref_data.shape = self.reference_dims[::-1]

        if self.is_mosaic:
            ref_data = np.tile(ref_data, self.mosaic_dims[::-1])
        return ref_data


    def get_output(
        self,
        load: bool = False,
        shifts: bool = True,
        flip: bool = False,
        clear_images: bool = True,
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
        if shifts and self.has_shifts:
            images = self.apply_shifts_to_images(images)
        if clear_images:
            self.clear_images()
            self.clear_reference()
        if not flip:
            # The default state is flipped with respect to how it's displayed in XRM Data Explorer
            return np.flip(images, axis=1)
        return images
