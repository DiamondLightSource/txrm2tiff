import unittest
from unittest.mock import MagicMock
from numpy.testing import assert_array_equal

import numpy as np

from txrm2tiff.txrm.shifts_mixin import ShiftsMixin


class TestShifts(unittest.TestCase):
    def test_apply_shifts_to_images_x_shift(self):
        base = range(5)
        arr = np.vstack([base, base, base, base])[np.newaxis, :, :]

        shifter = ShiftsMixin()
        shifter.read_stream = MagicMock(side_effect=[[2], [0]])
        output = shifter.apply_shifts_to_images(images=arr)

        expected_base = [3, 4, 0, 1, 2]
        expected_output = np.vstack(
            [expected_base, expected_base, expected_base, expected_base]
        )[np.newaxis, :, :]
        assert_array_equal(output, expected_output)

    def test_apply_shifts_to_images_y_shift(self):
        base = range(5)
        arr = np.vstack([base, base, base, base]).transpose()[np.newaxis, :, :]

        shifter = ShiftsMixin()
        shifter.read_stream = MagicMock(side_effect=[[0], [2]])
        output = shifter.apply_shifts_to_images(images=arr)

        expected_base = [3, 4, 0, 1, 2]
        expected_output = np.vstack(
            [expected_base, expected_base, expected_base, expected_base]
        ).transpose()[np.newaxis, :, :]
        assert_array_equal(output, expected_output)

    def test_apply_shifts_to_images_xy_shift(self):
        base = np.arange(0, 5)
        width = len(base)
        arr = np.vstack(
            [
                base,
                base + width,
                base + 2 * width,
                base + 3 * width,
                base + 4 * width,
                base + 5 * width,
            ]
        )[np.newaxis, :, :]

        shifter = ShiftsMixin()
        shifter.read_stream = MagicMock(side_effect=[[3], [2]])
        output = shifter.apply_shifts_to_images(images=arr)

        expected_output = np.asarray(
            [
                [
                    [22, 23, 24, 20, 21],
                    [27, 28, 29, 25, 26],
                    [2, 3, 4, 0, 1],
                    [7, 8, 9, 5, 6],
                    [12, 13, 14, 10, 11],
                    [17, 18, 19, 15, 16],
                ]
            ]
        )
        assert_array_equal(output, expected_output)
