import logging
from typing import List
import numpy as np

from .txrm_property import txrm_property


class ShiftsMixin:
    @txrm_property(fallback=None)
    def has_shifts(self) -> bool:
        return (
            self.has_stream("Alignment/X-Shifts")
            and self.has_stream("Alignment/Y-Shifts")
            and self.shifts_applied
        )

    @txrm_property(fallback=None)
    def shifts_applied(self):
        return np.any(self.x_shifts) or np.any(self.y_shifts)

    @txrm_property(fallback=None)
    def x_shifts(self) -> List:
        return self.read_stream("Alignment/X-Shifts")

    @txrm_property(fallback=None)
    def y_shifts(self) -> List:
        return self.read_stream("Alignment/Y-Shifts")

    def apply_shifts_to_images(self, images: np.ndarray) -> np.ndarray:
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
