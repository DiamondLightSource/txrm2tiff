import unittest
from parameterized import parameterized
from unittest.mock import MagicMock, patch
from numpy.testing import assert_array_equal

import struct
import numpy as np

from txrm2tiff import txrm_functions
from txrm2tiff.xradia_properties.enums import XrmDataTypes


class TestTxrmFunctions(unittest.TestCase):
    def test_reading_string_stream(self):
        ole = MagicMock()
        stream = MagicMock()
        stream_name = "Test"
        ole.openstream.return_value = stream
        ole.exists.return_value = True
        stream.read.return_value = (
            b"Pixis\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        )

        stream_list = txrm_functions.general._read_text_stream_to_list(ole, stream_name)

        ole.openstream.assert_called_with(stream_name)

        self.assertEqual(stream_list, ["Pixis"], msg="output is not as expected")

    @parameterized.expand(
        [(x,) for x in XrmDataTypes if x != XrmDataTypes.XRM_STRING],
    )  # Strings are handled by _read_text_stream_to_list
    def test_read_XRM_data_types_from_stream(self, dtype):
        ole = MagicMock()
        stream = MagicMock()
        stream_name = f"Test_{dtype.name}"
        ole.openstream.return_value = stream
        ole.exists.return_value = True
        values = np.arange(0, 100).astype(dtype.value)
        stream.getvalue.return_value = values.tobytes()
        try:
            output = txrm_functions.general._read_number_stream_to_list(
                ole, stream_name, dtype=dtype.value
            )
        except Exception:
            print(f"Failed for {dtype}")
            raise
        self.assertEqual(output, values.tolist())

    @parameterized.expand(
        [(x,) for x in XrmDataTypes if x != XrmDataTypes.XRM_STRING],
    )  # Images are not made of strings
    @patch("txrm2tiff.txrm_functions.images.extract_image_dtype")
    def test_extracts_image_with_XRM_data_types(self, dtype, patched_dtype_extractor):
        im_shape = (6, 9)

        ole = MagicMock()
        ole.exists.return_value = True

        stream = MagicMock()
        ole.openstream.return_value = stream
        image = np.arange(np.prod(im_shape), dtype=dtype.value)
        image.shape = im_shape
        stream.getvalue.return_value = image.tobytes()
        patched_dtype_extractor.return_value = dtype
        output = txrm_functions.extract_single_image(
            ole, 24, im_shape[0], im_shape[1], strict=True
        )

        assert_array_equal(
            image, output, err_msg=f"Failed to unpack image with dtype={dtype}"
        )

    @parameterized.expand(
        [(XrmDataTypes.XRM_UNSIGNED_SHORT,), (XrmDataTypes.XRM_FLOAT,)]
    )
    @patch("txrm2tiff.txrm_functions.images.extract_image_dtype")
    def test_extracting_single_image_unknown_type(self, dtype, patched_dtype_extractor):
        im_shape = (6, 9)

        ole = MagicMock()
        ole.exists.return_value = True

        stream = MagicMock()
        ole.openstream.return_value = stream
        image = np.arange(np.prod(im_shape), dtype=dtype.value)
        image.shape = im_shape
        stream.getvalue.return_value = image.tobytes()
        patched_dtype_extractor.return_value = None
        output = txrm_functions.extract_single_image(
            ole, 24, im_shape[0], im_shape[1], strict=True
        )

        assert_array_equal(
            image, output, err_msg=f"Failed to unpack image with dtype={dtype}"
        )

    def test_extracting_single_image_short(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        ole.exists.side_effect = [True, False]
        stream.getvalue.return_value = struct.pack("<6H", *list(range(6)))

        data = txrm_functions.extract_single_image(ole, 1, 2, 3)

        ole.openstream.assert_called_with("ImageData1/Image1")

        assert_array_equal(
            data, np.arange(6).reshape(2, 3), err_msg="output is not as expected"
        )

    def test_extracting_single_image_float(self):
        ole = MagicMock()
        stream = MagicMock()
        ole.openstream.return_value = stream
        ole.exists.side_effect = [True, False]
        stream.getvalue.return_value = struct.pack("<6f", *list(range(6)))

        data = txrm_functions.extract_single_image(ole, 1, 2, 3)

        ole.openstream.assert_called_with("ImageData1/Image1")

        assert_array_equal(
            data, np.arange(6).reshape(2, 3), err_msg="output is not as expected"
        )

    def test_read_stream_failure(self):
        ole = MagicMock()
        ole.exists.return_value = False
        data = txrm_functions.read_stream(ole, "key", "i")

        self.assertFalse(data)


if __name__ == "__main__":
    unittest.main()
