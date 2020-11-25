import unittest
from unittest.mock import MagicMock, patch
from numpy.testing import assert_array_equal

import struct
import numpy as np
from scipy.constants import h, c, e

from txrm2tiff import txrm_wrapper


def create_ole_that_returns_integer():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream

    packed_pixel_size = struct.pack('<I', 100)
    stream.getvalue.side_effect = [packed_pixel_size]
    return ole


def create_ole_that_returns_float():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream

    packed_pixel_size = struct.pack('<f', 100.5)
    stream.getvalue.side_effect = [packed_pixel_size]
    return ole


def pack_int(number):
    return struct.pack('<I', np.int(number))

class TestTxrmWrapper(unittest.TestCase):
    def test_extracting_single_image_short(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        ole.exists.side_effect = [True, False]
        stream.getvalue.return_value = struct.pack('<6H', *list(range(6)))

        data = txrm_wrapper.extract_single_image(ole, 1, 2, 3)

        ole.openstream.assert_called_with('ImageData1/Image1')

        assert_array_equal(data, np.arange(6).reshape(2, 3), err_msg="output is not as expected")

    def test_extracting_single_image_float(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        ole.exists.side_effect = [True, False]
        stream.getvalue.return_value = struct.pack('<6f', *list(range(6)))

        data = txrm_wrapper.extract_single_image(ole, 1, 2, 3)

        ole.openstream.assert_called_with('ImageData1/Image1')

        assert_array_equal(data, np.arange(6).reshape(2, 3), err_msg="output is not as expected")

    def test_extracting_single_image_unexpected_type(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        ole.exists.side_effect = [True, False]
        stream.getvalue.return_value = struct.pack('<6q', *list(range(6)))

        with self.assertRaises(TypeError):
            data = txrm_wrapper.extract_single_image(ole, 1, 2, 3)
            ole.openstream.assert_called_with('ImageData1/Image1')
            ole.openstream.assert_called_with('ImageInfo/DataType')

    def test_read_stream_failure(self):
        ole = MagicMock()
        ole.exists.return_value = False
        data = txrm_wrapper.read_stream(ole, "key", 'i')

        self.assertIsNone(data)


    def test_extracts_dimensions(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        stream.getvalue.side_effect = [pack_int(i) for i in [6, 7]]

        data = txrm_wrapper.extract_image_dims(ole)

        self.assertListEqual(data, [6, 7])


    def test_extracts_number_of_images(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        stream.getvalue.side_effect = [pack_int(i) for i in [9]]

        data = txrm_wrapper.extract_number_of_images(ole)

        self.assertEqual(data, 9)


    def test_extracts_multiple_images(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.exists.side_effect = ([True] * 5) + ([False, True] * 5) + [False]
        ole.openstream.return_value = stream
        dimensions = [pack_int(i) for i in [2, 3]]
        images = [struct.pack('<6H', *list(range(6)))] * 5
        images_taken = [pack_int(5)]
        stream.getvalue.side_effect = dimensions + images_taken + images

        data = txrm_wrapper.extract_all_images(ole)

        self.assertEqual(len(data), 5)


    def test_extracts_multiple_images_with_dtypes(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.exists.side_effect = ([True] * 5) + ([True, True] * 5) + [False]
        ole.openstream.return_value = stream
        dimensions = [pack_int(i) for i in [2, 3]]
        dtype = pack_int(5)
        images_and_dtypes = [struct.pack('<6H', *list(range(6))), dtype] * 5
        images_taken = [pack_int(5)]
        stream.getvalue.side_effect = dimensions + images_taken + images_and_dtypes

        data = txrm_wrapper.extract_all_images(ole)

        self.assertEqual(len(data), 5)



    def test_extracts_tilt_angles(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        packed_tilts = struct.pack('<4f', 1, 2, 3, 4)
        stream.getvalue.side_effect = [packed_tilts]

        data = txrm_wrapper.extract_tilt_angles(ole)
        ole.openstream.assert_called_with('ImageInfo/Angles')
        assert_array_equal(data, np.array([1, 2, 3, 4]))


    def test_extacts_exposure_time_tomo(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        # Testing uneven tilt (more +ve than -ve values):
        packed_exposures = struct.pack('<9f', 1, 2, 3, 4, 5, 6, 7, 8, 9)
        packed_angles = struct.pack('<9f', -3, -2, -1, 0, 1, 2, 3, 4, 5)
        stream.getvalue.side_effect = [packed_exposures, packed_angles]
        ole.exists.return_value = True
        data = txrm_wrapper.extract_exposure_time(ole)

        self.assertEqual(data, 4.)


    def test_extacts_exposure_time_single_image(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        packed_exposure = struct.pack('<1f', 5)
        stream.getvalue.side_effect = [packed_exposure]
        ole.exists.side_effect = [False, True, True]
        data = txrm_wrapper.extract_exposure_time(ole)

        self.assertEqual(data, 5)


    def test_extacts_pixel_size(self):
        ole = create_ole_that_returns_float()
        data = txrm_wrapper.extract_pixel_size(ole)

        self.assertEqual(data, 100.5)


    def test_extracts_xray_magnification(self):
        ole = create_ole_that_returns_float()
        data = txrm_wrapper.extract_xray_magnification(ole)

        self.assertEqual(data, 100.5)


    def test_extracts_energy(self):
        ole = create_ole_that_returns_float()
        data = txrm_wrapper.extract_energy(ole)

        self.assertEqual(data, 100.5)


    def test_extracts_wavelength(self):
        ole = create_ole_that_returns_float()
        data = txrm_wrapper.extract_wavelength(ole)
        self.assertAlmostEqual(float("%.8e" % data), 1.23367361e-8)


    def test_create_mosaic_of_reference_image(self):
        reference_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        mosaic_reference = txrm_wrapper.create_reference_mosaic(MagicMock(), reference_data, 6, 3, 2, 1)
        expected_reference = np.array([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9],
                                    [1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9]
                                    ])
        self.assertEqual(mosaic_reference.shape, expected_reference.shape, msg="Arrays must be the same shape")
        assert_array_equal(mosaic_reference, expected_reference)


    def test_rescale_ref_exposure(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        # Testing uneven tilt (more +ve than -ve values):
        ref_exposure = struct.pack('<1f', 2)
        packed_exposures = struct.pack('<9f', 1, 2, 3, 4, 5, 6, 7, 8, 9)
        packed_angles = struct.pack('<9f', -3, -2, -1, 0, 1, 2, 3, 4, 5)
        stream.getvalue.side_effect = [ref_exposure, packed_exposures, packed_angles]
        ole.exists.return_value = True
        # In this case, the values should be returned as doubled due to the reference
        # exposure being half of the central image exposure
        assert_array_equal(
            txrm_wrapper.rescale_ref_exposure(ole, np.array([2., 4., 6.])),
            np.array([4., 8., 12.]))

if __name__ == '__main__':
    unittest.main()
