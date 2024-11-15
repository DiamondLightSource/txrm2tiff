from datetime import datetime
import unittest
from unittest.mock import MagicMock, patch
from numpy.testing import assert_array_equal

import numpy as np

from txrm2tiff.txrm.abstract import AbstractTxrm
from txrm2tiff.xradia_properties.enums import XrmDataTypes


@patch.multiple(
    AbstractTxrm, __abstractmethods__=set(), open=MagicMock
)  # Allows abstract method to be initialised and file to not be opened
class AbstractTxrmTest(unittest.TestCase):
    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.image_info", new={"ImagesTaken": [500]}
    )
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.shape", new=(6, 9))
    def test_extract_images(self):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        images = [
            np.arange(a, a + np.prod(txrm.shape))
            for a in np.arange(txrm.image_info["ImagesTaken"][0])
        ]
        for im in images:
            im.shape = txrm.shape
        with patch.object(txrm, "_extract_single_image") as mocked_extract_single_image:
            mocked_extract_single_image.side_effect = images
            output = txrm.extract_images()

        expected = np.asarray(images)

        assert_array_equal(output, expected)

    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.image_info", new={"ImagesTaken": [500]}
    )
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.shape", new=(6, 9))
    def test_extract_images_no_images(self):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        with patch.object(txrm, "_extract_single_image") as mocked_extract_single_image:
            mocked_extract_single_image.side_effect = TypeError("Test")
            output = txrm.extract_images()

        self.assertTrue(output.size == 0)

    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.image_info", new={"ImagesTaken": [500]}
    )
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.shape", new=(6, 9))
    def test_extract_images_no_images_strict(self):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        txrm.strict = True
        with patch.object(txrm, "_extract_single_image") as mocked_extract_single_image:
            mocked_extract_single_image.side_effect = TypeError("Test")
            with self.assertRaises(TypeError):
                txrm.extract_images()

    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.shape", new=(6, 9))
    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.image_dtype",
        XrmDataTypes.XRM_UNSIGNED_INT,
    )
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.has_stream")
    @patch("txrm2tiff.txrm.abstract.txrm_functions.get_stream_from_bytes")
    def test_extract_single_image_known_dtype(
        self, mocked_frombytes, mocked_has_stream
    ):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        image = np.arange(np.prod(txrm.shape))
        stream_data = image.tobytes()
        stream = MagicMock()
        stream.getvalue.return_value = stream_data
        txrm.ole = MagicMock()
        txrm.ole.openstream.return_value = stream

        mocked_frombytes.return_value = image
        output = txrm._extract_single_image(201)
        mocked_frombytes.assert_called_once_with(
            stream_data, dtype=txrm.image_dtype.value
        )
        mocked_has_stream.assert_called_once_with("ImageData3/Image201")
        image.shape = txrm.shape
        assert_array_equal(output, image)

    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.shape", new=(6, 9))
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.has_stream")
    @patch("txrm2tiff.txrm.abstract.txrm_functions.fallback_image_interpreter")
    def test_extract_single_image_unknown_dtype(
        self,
        mocked_interpreter,
        mocked_has_stream,
    ):
        mocked_has_stream.return_value = True
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        imsize = np.prod(txrm.shape)
        image = np.arange(imsize)
        stream_data = image.tobytes()
        stream = MagicMock()
        stream.getvalue.return_value = stream_data
        txrm.ole = MagicMock()
        txrm.ole.openstream.return_value = stream

        mocked_interpreter.return_value = image
        output = txrm._extract_single_image(201)
        mocked_interpreter.assert_called_once_with(stream_data, imsize, txrm.strict)
        mocked_has_stream.assert_called_once_with("ImageData3/Image201")
        image.shape = txrm.shape
        assert_array_equal(output, image)

    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.shape", new=(6, 9))
    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.image_dtype",
        XrmDataTypes.XRM_UNSIGNED_INT,
    )
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.has_stream")
    @patch("txrm2tiff.txrm.abstract.txrm_functions.get_stream_from_bytes")
    @patch("txrm2tiff.txrm.abstract.txrm_functions.fallback_image_interpreter")
    def test_extract_single_image_fails(
        self,
        mocked_interpreter,
        mocked_frombytes,
        mocked_has_stream,
    ):
        mocked_has_stream.return_value = True
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        imsize = np.prod(txrm.shape)
        image = np.arange(imsize)
        stream_data = image.tobytes()
        stream = MagicMock()
        stream.getvalue.return_value = stream_data
        txrm.ole = MagicMock()
        txrm.ole.openstream.return_value = stream

        empty_img = np.asarray([])

        mocked_frombytes.side_effect = TypeError("Test")
        mocked_interpreter.return_value = empty_img
        output = txrm._extract_single_image(201)
        mocked_interpreter.assert_called_once_with(stream_data, imsize, txrm.strict)
        mocked_has_stream.assert_called_once_with("ImageData3/Image201")
        image.shape = txrm.shape
        assert_array_equal(output, empty_img)

    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.get_images")
    def test_get_single_image_is_loaded(self, mocked_get_images):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        img = "test_img"
        images = [img]
        mocked_get_images.return_value = images
        output = txrm.get_single_image(1)
        mocked_get_images.assert_called_once_with(load=False)
        self.assertEqual(output, img)

    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.get_images")
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm._extract_single_image")
    def test_get_single_image_is_not_loaded(
        self, mocked_extract_single, mocked_get_images
    ):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        img = "test_img"
        mocked_extract_single.return_value = img
        mocked_get_images.return_value = None
        output = txrm.get_single_image(1)
        mocked_get_images.assert_called_once_with(load=False)
        self.assertEqual(output, img)

    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.reference_dtype",
        XrmDataTypes.XRM_UNSIGNED_INT,
    )
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.has_reference", new=True)
    @patch("txrm2tiff.txrm.abstract.txrm_functions.get_stream_from_bytes")
    def test_extract_reference_data(self, mocked_frombytes):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        stream = MagicMock()
        stream.getvalue.return_value = "ref_data"
        txrm.ole = MagicMock()
        txrm.ole.openstream.return_value = stream

        mocked_frombytes.return_value = "output"
        output = txrm.extract_reference_data()
        mocked_frombytes.assert_called_once_with(
            "ref_data", dtype=txrm.reference_dtype.value
        )
        self.assertEqual(output, "output")

    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.has_reference", new=True)
    @patch("txrm2tiff.txrm.abstract.txrm_functions.get_stream_from_bytes")
    def test_extract_reference_data_unknown_dtype_returns_bytes(self, mocked_frombytes):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        stream = MagicMock()
        stream.getvalue.return_value = "ref_data"  # This would be bytes in a real case
        txrm.ole = MagicMock()
        txrm.ole.openstream.return_value = stream
        output = txrm.extract_reference_data()
        mocked_frombytes.assert_not_called()
        self.assertEqual(output, "ref_data")

    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.image_info", new={"ImagesTaken": [97]})
    def test_get_central_image(self):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        with patch.object(txrm, "get_single_image") as mocked_get_single_image:
            txrm.get_central_image()
            mocked_get_single_image.assert_called_once_with(49)

    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.image_info",
        new={"ImagesTaken": [500]},
    )
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.is_mosaic", new=False)
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.image_dims", new=[6, 7])
    def test_output_shape_non_mosaic(self):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        self.assertEqual(txrm.output_shape, [500, 7, 6])

    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.image_info",
        new={"ImagesTaken": [6], "MosiacColumns": [2], "MosiacRows": [3]},
    )
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.is_mosaic", new=True)
    @patch("txrm2tiff.txrm.abstract.AbstractTxrm.image_dims", new=[6, 7])
    def test_output_shape_mosaic(self):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        self.assertEqual(txrm.output_shape, [1, 7 * 3, 6 * 2])

    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.image_info",
        new={"Date": ["ksfs$oio12/30/2021 23:55:59!j#f"]},
    )
    def test_datetimes(self):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        self.assertEqual(txrm.datetimes[0], datetime(2021, 12, 30, 23, 55, 59))

    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.image_info",
        new={"Date": ["ksfs$oio12/30/21 23:55:59!j#f"]},
    )
    def test_datetimes_without_century(self):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        self.assertEqual(txrm.datetimes[0], datetime(2021, 12, 30, 23, 55, 59))

    @patch(
        "txrm2tiff.txrm.abstract.AbstractTxrm.image_info",
        new={"Date": ["ksfs$oio12/30/2021 23:55:59.83!j#f"]},
    )
    def test_datetimes_with_milliseconds(self):
        txrm = AbstractTxrm("test/path", load_images=False, load_reference=False)
        self.assertEqual(txrm.datetimes[0], datetime(2021, 12, 30, 23, 55, 59, 830000))
