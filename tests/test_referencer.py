from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, MagicMock
from parameterized import parameterized
from numpy.testing import assert_array_equal

from pathlib import Path
import numpy as np
from oxdls import OMEXML

from txrm2tiff.utils.file_handler import manual_save
from txrm2tiff.txrm.ref_mixin import ReferenceMixin
from txrm2tiff.txrm.v3 import Txrm3
from txrm2tiff.txrm.v5 import Txrm5


class TestReferencer(unittest.TestCase):
    def test_apply_reference_defaults_with_internal(self):
        referencer = ReferenceMixin()
        referencer.has_reference = True
        with patch.object(
            referencer, "apply_internal_reference"
        ) as mocked_apply_internal:
            referencer.apply_reference()
            mocked_apply_internal.assert_called_once_with(
                overwrite=True, compensate_exposure=True
            )

    def test_apply_reference_defaults_no_internal(self):
        referencer = ReferenceMixin()
        referencer.has_reference = False
        with patch.object(
            referencer, "apply_internal_reference"
        ) as mocked_apply_internal:
            referencer.apply_reference()
            mocked_apply_internal.assert_not_called()

    @patch("txrm2tiff.txrm.ref_mixin.file_can_be_opened", MagicMock(return_value=True))
    def test_apply_reference_tiff(self):
        referencer = ReferenceMixin()
        with patch.object(
            referencer, "apply_custom_reference_from_array"
        ) as mocked_apply_array:
            with TemporaryDirectory("tiff_ref_test_", dir=".") as tmp_dir:
                array = np.ones((3, 5, 5))
                tif_path = Path(tmp_dir) / "image.ome.tiff"
                manual_save(tif_path, array)

                referencer.apply_reference(tif_path, False)

        args = mocked_apply_array.call_args[0]
        kwargs = mocked_apply_array.call_args[1]
        assert_array_equal(args[0], array)
        self.assertEqual(kwargs, {"overwrite": True})

    @patch("txrm2tiff.txrm.ref_mixin.file_can_be_opened", MagicMock(return_value=True))
    def test_apply_reference_tiff_compensate_exposure(self):
        referencer = ReferenceMixin()
        with patch.object(
            referencer, "apply_custom_reference_from_array"
        ) as mocked_apply_array:
            with TemporaryDirectory("tiff_ref_test_", dir=".") as tmp_dir:
                array = np.ones((3, 5, 5))
                exposures = (1, 2, 3)
                metadata = OMEXML()
                metadata.image().Pixels.set_plane_count(len(exposures))
                for idx, exposure in enumerate(exposures):
                    metadata.image().Pixels.Plane(idx).set_ExposureTime(exposure)
                tif_path = Path(tmp_dir) / "image.ome.tiff"
                manual_save(tif_path, array, metadata=metadata)

                referencer.apply_reference(tif_path, True)
        args = mocked_apply_array.call_args[0]
        kwargs = mocked_apply_array.call_args[1]
        assert_array_equal(args[0], array)
        self.assertEqual(
            kwargs, {"overwrite": True, "custom_exposure": np.mean(exposures)}
        )

    @parameterized.expand([(Txrm3,), (Txrm5,)])
    @patch("txrm2tiff.txrm.ref_mixin.file_can_be_opened", MagicMock(return_value=True))
    @patch("txrm2tiff.txrm.ref_mixin.isOleFile", MagicMock(return_value=True))
    @patch("txrm2tiff.txrm.ref_mixin.main.open_txrm")
    def test_apply_refererence_txrm(self, TxrmClass, mocked_open_txrm):
        referencer = ReferenceMixin()
        txrm = MagicMock(auto_spec=TxrmClass)
        mocked_open_txrm.return_value.__enter__.return_value = txrm

        with patch.object(referencer, "apply_reference_from_txrm") as mocked_from_txrm:
            referencer.apply_reference("test/path.txrm")
            mocked_from_txrm.assert_called_once_with(txrm)
