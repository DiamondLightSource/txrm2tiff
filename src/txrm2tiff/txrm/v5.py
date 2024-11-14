from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
import typing

from .abc.ome import TxrmWithOME
from .annotator import Annotator
from .wrappers import txrm_property
from ..utils.image_processing import stitch_images

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray, DTypeLike
    from os import PathLike
    from io import IOBase


class Txrm5(TxrmWithOME):

    def __init__(
        self,
        f: str | PathLike[typing.Any] | IOBase | bytes,
        /,
        load_images: bool = True,
        load_reference: bool = True,
        strict: bool = False,
    ) -> None:
        self._annotator = Annotator(self)
        super().__init__(
            f, load_images=load_images, load_reference=load_reference, strict=strict
        )

    @txrm_property(fallback=None)
    def is_mosaic(self) -> bool:
        if self.mosaic_dims is None:
            return False
        return bool(np.sum(self.mosaic_dims))

    @txrm_property(fallback=None)
    def image_dims(self) -> tuple[int, int]:
        return (
            self.image_info["ImageWidth"][0],
            self.image_info["ImageHeight"][0],
        )

    @txrm_property(fallback=None)
    def reference_dims(self) -> tuple[int, int]:
        return (
            self.reference_info["ImageWidth"][0],
            self.reference_info["ImageHeight"][0],
        )

    @txrm_property(fallback=None)
    def reference_exposure(self) -> typing.Optional[float] | None:
        if "ExpTimes" in self.reference_info:
            return self.reference_info["ExpTimes"][0]
        elif "ExpTime" in self.reference_info:
            return self.reference_info["ExpTime"][0]
        return None

    def get_output(
        self,
        load: bool = False,
        shifts: bool = False,
        flip: bool = False,
        clear_images: bool = True,
    ) -> typing.Optional[NDArray[typing.Any]]:
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
        if self.is_mosaic and self.mosaic_dims is not None:
            images = stitch_images(images, self.mosaic_dims)
        if clear_images:
            self.clear_images()
            self.clear_reference()
        if not flip:
            # The default state is flipped with respect to how it's displayed in XRM Data Explorer
            return np.flip(images, axis=1)
        return images

    def annotate(
        self, scale_bar: bool = True, clip_percentiles: tuple[float, float] = (2, 98)
    ) -> NDArray[typing.Any] | None:
        return self._annotator.annotate(
            scale_bar=scale_bar, clip_percentiles=clip_percentiles
        )

    def save_images(
        self,
        filepath: str | PathLike[str] | None = None,
        datatype: DTypeLike | None = None,
        shifts: bool = False,
        flip: bool = False,
        clear_images: bool = False,
        mkdir: bool = False,
        strict: bool | None = None,
        save_annotations: bool = True,
        annotated_path: str | PathLike[str] | None = None,
    ) -> None:
        if strict is None:
            strict = self.strict

        filepath = super().save_images(
            filepath=filepath,
            datatype=datatype,
            shifts=shifts,
            flip=flip,
            clear_images=clear_images,
            mkdir=mkdir,
            strict=strict,
        )
        if filepath is not None and save_annotations:
            if annotated_path is None:
                # Generate default path
                filename = filepath.name
                if filename.lower().endswith(".ome.tiff"):
                    # Special case for ome.tiff
                    stem, suffix = filename.rsplit(".ome.", 1)
                else:
                    stem, suffix = filename.rsplit(".", 1)
                annotated_path = filepath.parent / f"{stem}_Annotated.{suffix}"

            self._annotator.save(annotated_path, mkdir=mkdir, strict=strict)
