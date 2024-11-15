from __future__ import annotations
import typing
import logging
import numpy as np

from .abc.ome import TxrmWithOME

from .wrappers import txrm_property
from ..utils.functions import convert_to_int

if typing.TYPE_CHECKING:
    from os import PathLike
    from io import IOBase
    from numpy.typing import NDArray


class Txrm3(TxrmWithOME):
    def __init__(
        self,
        f: str | PathLike[typing.Any] | IOBase | bytes,
        /,
        load_images: bool = True,
        load_reference: bool = True,
        strict: bool = True,
    ) -> None:
        super().__init__(
            f,
            load_images=load_images,
            load_reference=load_reference,
            strict=strict,
        )

    @txrm_property(fallback=None)
    def is_mosaic(self) -> bool:
        return self.image_info.get("MosiacMode", [0])[0] == 1

    @txrm_property(fallback=None)
    def image_dims(self) -> tuple[int, int]:
        img_dims = (
            int(self.image_info["ImageWidth"][0]),
            int(self.image_info["ImageHeight"][0]),
        )
        if self.is_mosaic:
            img_dims = self._mosaic_to_single_image_dims(img_dims)
        return img_dims

    @txrm_property(fallback=None)
    def reference_dims(self) -> tuple[int, int]:
        dims = self.image_dims
        assert dims is not None
        return dims  # Same for both (reference width/height isn't stored separately for v3)

    def _mosaic_to_single_image_dims(self, dims) -> tuple[int, int]:
        mosaic_dims = self.mosaic_dims
        assert mosaic_dims is not None
        return (
            convert_to_int(dims[0] / mosaic_dims[0]),
            convert_to_int(dims[1] / mosaic_dims[1]),
        )

    @txrm_property(fallback=None)
    def reference_exposure(self) -> float | None:
        return typing.cast(
            float | None, self.read_single_value_from_stream("ReferenceData/ExpTime")
        )

    def load_reference(self) -> None:
        super().load_reference()
        if (
            self._reference is not None  # type: ignore[has-type]
            and self.is_mosaic
            and self.mosaic_dims is not None
        ):
            self._reference = np.tile(
                self._reference, self.mosaic_dims[::-1]  # type: ignore[has-type]
            )

    def get_output(
        self,
        load: bool = False,
        shifts: bool = False,
        flip: bool = False,
        clear_images: bool = True,
    ) -> NDArray[typing.Any] | None:
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
