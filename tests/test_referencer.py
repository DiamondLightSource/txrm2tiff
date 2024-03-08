from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, MagicMock
from parameterized import parameterized
from numpy.testing import assert_array_equal

from pathlib import Path
import numpy as np
from ome_types import model
from ome_types.model.simple_types import PixelType
from ome_types.model.pixels import DimensionOrder


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
                exposures = tuple(range(1, array.shape[0] + 1))
                metadata = model.OME(
                    images=[
                        model.Image(
                            id="Image:0",
                            pixels=model.Pixels(
                                dimension_order=DimensionOrder.XYCTZ,
                                id="Pixels:0",
                                size_c=1,
                                size_t=1,
                                size_x=array.shape[2],
                                size_y=array.shape[1],
                                size_z=array.shape[0],
                                type=PixelType.FLOAT,
                                tiff_data_blocks=[
                                    model.TiffData(first_z=i, ifd=i)
                                    for i in range(array.shape[0])
                                ],
                                planes=[
                                    model.Plane(
                                        the_c=0, the_t=0, the_z=i, exposure_time=exp
                                    )
                                    for i, exp in enumerate(exposures)
                                ],
                            ),
                        ),
                    ]
                )
                tif_path = Path(tmp_dir) / "image.ome.tif"
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
