import unittest

from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile, TemporaryDirectory
import numpy as np
from numpy.testing import assert_array_equal
from oxdls import OMEXML
import tifffile as tf

from txrm2tiff.utils.file_handler import (
    file_can_be_opened,
    ole_file_works,
    manual_save,
    manual_annotation_save,
)


class TestFileHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_temp = NamedTemporaryFile(suffix=".txrm", delete=False)
        cls.fake_file = Path(cls.fake_temp.name)
        cls.real_file = Path(
            "/dls/science/groups/das/ExampleData/B24_test_data/data/2019/cm98765-1/raw/XMv10/test_tomo2_e3C_full.txrm"
        )

    @classmethod
    def tearDownClass(cls):
        cls.fake_temp.file.close()
        cls.fake_file.unlink()

    def test_ole_file_returns_true_for_real_file(self):
        if self.real_file.exists():
            self.assertTrue(ole_file_works(self.real_file))
        else:
            print("Cannot run test without access to dls directories")

    def test_real_file_can_be_opened(self):
        if self.real_file.exists():
            self.assertTrue(file_can_be_opened(self.real_file))
        else:
            print("Cannot run test without access to dls directories")

    def test_ole_file_returns_false_for_fake_file(self):
        self.assertFalse(ole_file_works(self.fake_file))

    def test_fake_file_can_be_opened(self):
        self.assertTrue(file_can_be_opened(self.fake_file))

    def test_nonexistent_file_can_be_opened(self):
        self.assertFalse(
            file_can_be_opened(
                Path(
                    "/fake/path/oh/no/{}".format(datetime.now().strftime("%Y%m%d_%H%M"))
                )
            )
        )

    def test_manual_save(self):
        with TemporaryDirectory(
            prefix="saving_test_", dir=Path(__name__).parent
        ) as tmpdir:
            im_path = Path(tmpdir) / "saved.tiff"
            image = np.zeros((5, 30, 30), dtype=np.float64)
            self.assertFalse(im_path.exists())
            manual_save(im_path, image)
            self.assertTrue(im_path.exists())

    def test_manual_save_with_datatype(self):
        target_dtype = np.uint16
        with TemporaryDirectory(
            prefix="saving_test_", dir=Path(__name__).parent
        ) as tmpdir:
            im_path = Path(tmpdir) / "saved.tiff"
            image = np.ones((5, 30, 35), dtype=np.float64)
            self.assertFalse(im_path.exists())
            manual_save(im_path, image, data_type=target_dtype)
            self.assertTrue(im_path.exists())
            with tf.TiffFile(im_path) as tiff:
                saved_arr = tiff.asarray()

        self.assertEqual(saved_arr.dtype, target_dtype)
        assert_array_equal(saved_arr, image)

    def test_manual_save_with_metadata(self):
        target_dtype = np.uint16

        with TemporaryDirectory(
            prefix="saving_test_", dir=Path(__name__).parent
        ) as tmpdir:
            im_path = Path(tmpdir) / "saved.tiff"
            metadata = OMEXML()

            image = np.ones((5, 30, 35), dtype=np.float64)
            self.assertFalse(im_path.exists())
            manual_save(im_path, image, data_type=np.uint16, metadata=metadata)
            self.assertTrue(im_path.exists())
            with tf.TiffFile(im_path) as tiff:
                saved_arr = tiff.asarray()
                saved_meta = tiff.pages[0].description

        # Image name and data type should be set while being saved
        metadata.image().set_Name(im_path.name)
        metadata.image().Pixels.set_PixelType("uint16")

        self.assertEqual(saved_arr.dtype, target_dtype)
        assert_array_equal(saved_arr, image)
        self.assertEqual(saved_meta, metadata.to_xml().strip())  # Remove final newline

    def test_manual_save_sets_pixel_size(self):
        with TemporaryDirectory(
            prefix="saving_test_", dir=Path(__name__).parent
        ) as tmpdir:
            im_path = Path(tmpdir) / "saved.tiff"
            metadata = OMEXML()
            pixel_size_xy = (1, 2)  # nm
            pixels = metadata.image().Pixels
            pixels.set_PhysicalSizeX(pixel_size_xy[0])
            pixels.set_PhysicalSizeY(pixel_size_xy[1])
            image = np.ones((5, 30, 35), dtype=np.float64)
            self.assertFalse(im_path.exists())
            manual_save(im_path, image, data_type=np.uint16, metadata=metadata)
            self.assertTrue(im_path.exists())
            with tf.TiffFile(im_path) as tiff:
                saved_arr = tiff.asarray()
                x_resolution = tiff.pages[0].tags["XResolution"].value
                y_resolution = tiff.pages[0].tags["YResolution"].value
                resolution_unit = tiff.pages[0].tags["ResolutionUnit"].value

        assert_array_equal(saved_arr, image)
        self.assertEqual(x_resolution, (int(1.0e7), 1))
        self.assertEqual(y_resolution, (int(5.0e6), 1))
        self.assertEqual(resolution_unit, int(tf.RESUNIT.CENTIMETER))

    def test_manual_annotation_save(self):
        with TemporaryDirectory(
            prefix="annotation_saving_test_", dir=Path(__name__).parent
        ) as tmpdir:
            im_path = Path(tmpdir) / "saved.tiff"
            image = np.ones((5, 30, 35, 3))
            self.assertFalse(im_path.exists())
            manual_annotation_save(im_path, image)
            self.assertTrue(im_path.exists())
            with tf.TiffFile(im_path) as tiff:
                saved_arr = tiff.asarray()

        assert_array_equal(saved_arr, image)


if __name__ == "__main__":
    unittest.main()
