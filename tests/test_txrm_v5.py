import unittest
from unittest.mock import patch, MagicMock
from numpy.testing import assert_array_equal

import numpy as np

from txrm2tiff.txrm.v5 import Txrm5


@patch.multiple(Txrm5, open=MagicMock)
class TestTxrm5(unittest.TestCase):
    @patch("txrm2tiff.txrm.v5.AbstractTxrm.mosaic_dims", new=(2, 3))
    def test_is_mosaic_true(self):
        txrm = Txrm5("test/path/file.txrm")
        self.assertTrue(txrm.is_mosaic)

    @patch("txrm2tiff.txrm.v5.AbstractTxrm.mosaic_dims", new=(0, 0))
    def test_is_mosaic_false(self):
        txrm = Txrm5("test/path/file.txrm")
        self.assertFalse(txrm.is_mosaic)

    def test_is_mosaic_false_if_no_mode(self):
        txrm = Txrm5("test/path/file.txrm")
        self.assertFalse(txrm.is_mosaic)

    @patch(
        "txrm2tiff.txrm.v5.AbstractTxrm.image_info",
        new={"ImageWidth": [20], "ImageHeight": [5]},
    )
    def test_image_dims(self):
        txrm = Txrm5("test/path/file.txrm")
        self.assertEqual(txrm.image_dims, [20, 5])

    @patch(
        "txrm2tiff.txrm.v5.AbstractTxrm.reference_info",
        new={"ImageWidth": [20], "ImageHeight": [5]},
    )
    def test_ref_image_dims(self):
        txrm = Txrm5("test/path/file.txrm")
        self.assertEqual(txrm.reference_dims, [20, 5])

    @patch("txrm2tiff.txrm.v5.AbstractTxrm.reference_info", new={"ExpTime": [0.5]})
    def test_reference_exposure(self):
        txrm = Txrm5("test/path/file.txrm")
        self.assertEqual(txrm.reference_exposure, 0.5)

    @patch("txrm2tiff.txrm.v5.Txrm5.is_mosaic", new=True)
    @patch("txrm2tiff.txrm.v5.Txrm5.reference_dims", new=[2, 3])
    @patch("txrm2tiff.txrm.v5.AbstractTxrm.extract_reference_data")
    @patch("txrm2tiff.txrm.v5.fallback_image_interpreter")
    def test_extract_reference_image_unknown_dtype(
        self, mocked_fallback_interpreter, mocked_extract_ref_data
    ):
        strict = False
        txrm = Txrm5(
            "test/path/file.txrm",
            load_images=False,
            load_reference=False,
            strict=strict,
        )
        ole = MagicMock()
        txrm.ole = ole
        im_size = np.prod(txrm.reference_dims)
        arr = np.arange(im_size, dtype=np.uint16)
        arr.shape = txrm.reference_dims[::-1]
        arr_flat = arr.flatten()
        arr_bytes = arr_flat.tobytes()
        mocked_extract_ref_data.return_value = arr_bytes
        mocked_fallback_interpreter.return_value = arr_flat

        output = txrm.extract_reference_image()

        mocked_extract_ref_data.assert_called_once_with()
        mocked_fallback_interpreter.assert_called_once_with(arr_bytes, im_size, strict)

        assert_array_equal(output, arr)

    @patch("txrm2tiff.txrm.v5.Txrm5.is_mosaic", new=True)
    @patch("txrm2tiff.txrm.v5.Txrm5.reference_dims", new=[2, 3])
    @patch("txrm2tiff.txrm.v5.AbstractTxrm.extract_reference_data")
    @patch("txrm2tiff.txrm.v5.fallback_image_interpreter")
    def test_extract_reference_image_known_dtype(
        self, mocked_fallback_interpreter, mocked_extract_ref_data
    ):
        strict = False
        txrm = Txrm5(
            "test/path/file.txrm",
            load_images=False,
            load_reference=False,
            strict=strict,
        )
        ole = MagicMock()
        txrm.ole = ole
        im_size = np.prod(txrm.reference_dims)
        arr = np.arange(im_size, dtype=np.uint16)
        arr.shape = txrm.reference_dims[::-1]
        arr_flat = arr.flatten()
        mocked_extract_ref_data.return_value = arr_flat

        output = txrm.extract_reference_image()

        mocked_extract_ref_data.assert_called_once_with()
        mocked_fallback_interpreter.assert_not_called()

        assert_array_equal(output, arr)
