import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_array_equal

from txrm2tiff.txrm.v3 import Txrm3


@patch.multiple(Txrm3, open=MagicMock)
class TestTxrm3(unittest.TestCase):
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.image_info", new={"MosiacMode": [1]})
    def test_is_mosaic_true(self):
        txrm = Txrm3("test/path/file.txrm")
        self.assertTrue(txrm.is_mosaic)

    @patch("txrm2tiff.txrm.v3.AbstractTxrm.image_info", new={"MosiacMode": [0]})
    def test_is_mosaic_false(self):
        txrm = Txrm3("test/path/file.txrm")
        self.assertFalse(txrm.is_mosaic)

    def test_is_mosaic_false_if_no_mode(self):
        txrm = Txrm3("test/path/file.txrm")
        self.assertFalse(txrm.is_mosaic)

    @patch(
        "txrm2tiff.txrm.v3.AbstractTxrm.image_info",
        new={"ImageWidth": [20], "ImageHeight": [5]},
    )
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.is_mosaic", new=True)
    def test_image_dims(self):
        txrm = Txrm3("test/path/file.txrm")
        self.assertEqual(txrm.image_dims, [20, 5])

    @patch(
        "txrm2tiff.txrm.v3.AbstractTxrm.image_info",
        new={"ImageWidth": [18], "ImageHeight": [14]},
    )
    @patch("txrm2tiff.txrm.v3.Txrm3.is_mosaic", new=True)
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.mosaic_dims", new=(6, 7))
    def test_image_dims_mosaic(self):
        txrm = Txrm3("test/path/file.txrm")
        self.assertEqual(txrm.image_dims, [3, 2])

    @patch(
        "txrm2tiff.txrm.v3.AbstractTxrm.image_info",
        new={"ImageWidth": [20], "ImageHeight": [5]},
    )
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.is_mosaic", new=True)
    def test_reference_dims(self):
        txrm = Txrm3("test/path/file.txrm")
        self.assertEqual(txrm.reference_dims, [20, 5])

    @patch(
        "txrm2tiff.txrm.v3.AbstractTxrm.image_info",
        new={"ImageWidth": [18], "ImageHeight": [14]},
    )
    @patch("txrm2tiff.txrm.v3.Txrm3.is_mosaic", new=True)
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.mosaic_dims", new=(6, 7))
    def test_reference_dims_mosaic(self):
        txrm = Txrm3("test/path/file.txrm")
        self.assertEqual(txrm.reference_dims, [3, 2])

    @patch("txrm2tiff.txrm.v3.AbstractTxrm.read_single_value_from_stream")
    def test_reference_exposure(self, mocked_read_single_value):
        txrm = Txrm3("test/path/file.txrm")
        mocked_read_single_value.return_value = "test"
        self.assertEqual(txrm.reference_exposure, "test")
        mocked_read_single_value.assert_called_once_with("ReferenceData/ExpTime")

    @patch("txrm2tiff.txrm.v3.Txrm3.is_mosaic", new=True)
    @patch("txrm2tiff.txrm.v3.Txrm3.reference_dims", new=[2, 3])
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.mosaic_dims", new=[1, 3])
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.extract_reference_data")
    @patch("txrm2tiff.txrm.v3.fallback_image_interpreter")
    def test_extract_reference_image_unknown_dtype(
        self, mocked_fallback_interpreter, mocked_extract_ref_data
    ):
        strict = False
        txrm = Txrm3(
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

        tiled_arr = np.vstack([arr] * 3)
        assert_array_equal(output, tiled_arr)

    @patch("txrm2tiff.txrm.v3.Txrm3.is_mosaic", new=True)
    @patch("txrm2tiff.txrm.v3.Txrm3.reference_dims", new=[2, 3])
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.mosaic_dims", new=[1, 3])
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.extract_reference_data")
    @patch("txrm2tiff.txrm.v3.fallback_image_interpreter")
    def test_extract_reference_image_known_dtype(
        self, mocked_fallback_interpreter, mocked_extract_ref_data
    ):
        strict = False
        txrm = Txrm3(
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

        tiled_arr = np.vstack([arr] * 3)
        assert_array_equal(output, tiled_arr)

    @patch("txrm2tiff.txrm.v3.AbstractTxrm.clear_images")
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.clear_reference")
    @patch("txrm2tiff.txrm.v3.AbstractTxrm.get_images")
    def test_get_output(
        self,
        mocked_get_images,
        mocked_clear_ref,
        mocked_clear_imgs,
    ):
        shape = (3, 4)
        arr = np.arange(np.prod(shape))
        arr.shape = shape
        mocked_get_images.return_value = arr

        strict = False
        txrm = Txrm3(
            "test/path/file.txrm",
            load_images=False,
            load_reference=False,
            strict=strict,
        )

        output = txrm.get_output(load=False, shifts=False, flip=True, clear_images=True)

        mocked_get_images.assert_called_once_with(False)
        mocked_clear_ref.assert_called_once_with()
        mocked_clear_imgs.assert_called_once_with()
        assert_array_equal(output, arr)
        self.assertNotEqual(id(output), id(arr))  # Should be a copy
