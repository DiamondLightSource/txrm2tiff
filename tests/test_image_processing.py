from random import randint
import unittest
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_approx_equal,
)

import numpy as np

from txrm2tiff.utils.image_processing import (
    cast_to_dtype,
    dynamic_despeckle_and_average_series,
    stitch_images,
    rescale_image,
)


class TestImageProcessing(unittest.TestCase):
    def test_stitch_mosaic(self):
        mosaic_xy_shape = (3, 4)
        image_size = (400, 400)
        images = np.zeros(
            (
                mosaic_xy_shape[0] * mosaic_xy_shape[1],
                image_size[1],
                image_size[0],
            )
        )
        range_array = np.repeat(
            np.arange(0, mosaic_xy_shape[1]), image_size[0] * image_size[1]
        )
        images.flat = list(range_array) * mosaic_xy_shape[0]

        output_image = stitch_images(images, mosaic_xy_shape)

        expected_array = np.concatenate(
            [
                np.concatenate(
                    [np.full(image_size, i) for i in range(0, mosaic_xy_shape[1])],
                    axis=1,
                )
                for _ in range(0, mosaic_xy_shape[0])
            ],
            axis=0,
        )[np.newaxis, :, :]

        assert_array_equal(output_image, expected_array)

    def test_despeckle_ave(self):
        custom_reference = []
        dims = (250, 250)
        speckle_per_frame = 5
        mid_point = 10
        for i in range(0, mid_point * 2 + 1):
            speckle_array = np.full(dims, i)
            for _ in range(speckle_per_frame):
                speckle_idx = (randint(0, dims[0] - 1), randint(0, dims[1] - 1))
                speckle_array[speckle_idx] = i * randint(
                    500, 1000
                )  # Should be well beyond 2.8 standard devs
            custom_reference.append(speckle_array)
        custom_reference = np.asarray(custom_reference)
        expected_approx_output = np.full(dims, mid_point, dtype=np.float32)
        with self.assertRaises(AssertionError):
            assert_array_almost_equal(
                custom_reference, expected_approx_output, decimal=0
            )  # Should not be almost equal until despeckle & averaging
        assert_array_almost_equal(
            dynamic_despeckle_and_average_series(custom_reference),
            expected_approx_output,
            decimal=0,
        )

    def test_rescale_image(self):
        original_minimum = -6000
        original_maximum = 6000
        step_size = 500

        target_minimum = -600
        target_maximum = 600
        expected_step_size = 50

        array = np.arange(original_minimum, original_maximum + step_size, step_size)
        output_array = rescale_image(
            array, target_minimum, target_maximum, original_minimum, original_maximum
        )
        self.assertEqual(output_array[0], target_minimum)
        self.assertEqual(output_array[-1], target_maximum)
        self.assertEqual(output_array[1] - output_array[0], expected_step_size)

    def test_rescale_image_keep_range(self):
        target_minimum = 0
        target_maximum = 255
        array = np.zeros((5, 5), np.uint8)
        array[0, 0] = 255

        output_array = rescale_image(array, target_minimum, target_maximum)
        self.assertTrue(np.all(output_array[1:] == target_minimum))
        self.assertEqual(output_array[0, 0], target_maximum)

    def test_cast_to_dtype_with_no_change(self):
        dtype = np.float64
        info = np.finfo(dtype)
        min, max = info.min, info.max
        image = np.asarray([min, max], dtype=dtype)
        output = cast_to_dtype(image, dtype)
        assert_array_equal(image, output)

    def test_cast_to_dtype_with_float64_to_uint16(self):
        dtype = np.float64
        info = np.finfo(dtype)
        min, max = info.min, info.max
        image = np.asarray([min, max], dtype=dtype)

        target_dtype = np.uint16
        target_info = np.iinfo(target_dtype)
        target_min, target_max = target_info.min, target_info.max
        output = cast_to_dtype(image, target_dtype)
        self.assertEqual(output.dtype, target_dtype)
        self.assertEqual(output.min(), target_min)
        self.assertEqual(output.max(), target_max)

    def test_cast_to_dtype_with_float64_to_float32(self):
        dtype = np.float64
        info = np.finfo(dtype)
        min, max = info.min, info.max
        image = np.asarray([min, max], dtype=dtype)
        target_dtype = np.float32
        target_info = np.finfo(target_dtype)
        target_min, target_max = target_info.min, target_info.max
        output = cast_to_dtype(image, target_dtype)
        self.assertEqual(output.dtype, target_dtype)
        self.assertEqual(output.min(), target_min)
        self.assertEqual(output.max(), target_max)

    def test_cast_to_dtype_with_int32_to_int16(self):
        dtype = np.int32
        info = np.iinfo(dtype)
        min, max = info.min, info.max
        count = 50
        image = np.linspace(min, max, count, endpoint=True, dtype=dtype)

        target_dtype = np.int16
        target_info = np.iinfo(target_dtype)
        target_min, target_max = target_info.min, target_info.max
        output = cast_to_dtype(image, target_dtype)
        self.assertEqual(output.dtype, target_dtype)
        self.assertEqual(output.min(), target_min)
        self.assertEqual(output.max(), target_max)

    def test_cast_to_dtype_with_int32_to_float32(self):
        dtype = np.int32
        info = np.iinfo(dtype)
        min, max = info.min, info.max
        count = 50
        image = np.linspace(min, max, count, endpoint=True, dtype=dtype)

        target_dtype = np.float32
        target_info = np.finfo(target_dtype)
        sig_figs = target_info.iexp

        output = cast_to_dtype(image, target_dtype)
        self.assertEqual(output.dtype, target_dtype)

        # Should only affect min and max significant figures
        assert_approx_equal(
            output.min(),
            min,
            significant=sig_figs,
            err_msg=f"Min is not equal to {sig_figs} sig figs",
        )
        assert_approx_equal(
            output.max(),
            max,
            significant=sig_figs,
            err_msg=f"Max is not equal to {sig_figs} sig figs",
        )

    def test_cast_to_dtype_with_int32_to_float64(self):
        dtype = np.int32
        info = np.iinfo(dtype)
        min, max = info.min, info.max
        count = 50
        image = np.linspace(min, max, count, endpoint=True, dtype=dtype)

        target_dtype = np.float64

        output = cast_to_dtype(image, target_dtype)
        self.assertEqual(output.dtype, target_dtype)
        self.assertEqual(output.min(), min)  # Should not affect min and max
        self.assertEqual(output.max(), max)
