import unittest
from unittest.mock import MagicMock, patch
from numpy.testing import assert_array_equal
from datetime import datetime

from parameterized import parameterized

import numpy as np
import os
import logging
import sys
from pathlib import Path
from functools import partial
from shutil import rmtree
from time import time, sleep
from olefile import OleFileIO

from txrm2tiff.txrm_to_image import TxrmToImage, _get_reference, _apply_reference, create_ome_metadata


class TestTxrmToImageSimple(unittest.TestCase):

    def test_divides_images_by_reference(self):
        num_images = 5
        images = [np.array([[0, 2, 4], [6, 8, 10]])] * num_images
        reference = np.arange(6).reshape(2, 3)
        resultant_images = _apply_reference(images, reference, np.uint16)
        self.assertEqual(len(resultant_images), num_images, msg="The result is the wrong length")
        expected_image = np.array([[0, 200, 200], [200, 200, 200]])
        for image in resultant_images:
            assert_array_equal(image, expected_image, err_msg="The result does not match the expected result")
 
    @patch('txrm2tiff.txrm_to_image.txrm_wrapper')
    def test_metadata_created_correctly(self, mocked_extractor):
        dims = (45, 40, 1)
        dtype = "uint16"
        test_image = [np.zeros(dims[:2], dtype=dtype)] * dims[2]
        mosaic_rows = 2
        mosaic_cols = 3

        ole = MagicMock()
        ole.exists.return_value = True
        mocked_extractor.extract_multiple_exposure_times.return_value = [2., 3., 4., 5., 0., 0.]
        mocked_extractor.extract_pixel_size.return_value = 0.005
        mocked_extractor.extract_x_coords.return_value = [7.5, 22.5, 37.5] * 2
        mocked_extractor.extract_y_coords.return_value = [10, 10, 10, 20, 20, 20]
        mocked_extractor.read_imageinfo_as_int.side_effect = [mosaic_rows, mosaic_cols]

        ome_metadata = create_ome_metadata(ole, test_image)
        self.assertEqual(ome_metadata.image().Pixels.get_SizeX(), dims[0])
        self.assertEqual(ome_metadata.image().Pixels.get_SizeY(), dims[1])
        self.assertEqual(ome_metadata.image().Pixels.get_SizeT(), dims[2])
        self.assertEqual(ome_metadata.image().Pixels.get_PixelType(), dtype)

    @patch('txrm2tiff.txrm_to_image.txrm_wrapper')
    def test_mosaic_exposure_averaged(self, mocked_extractor):
        dims = (45, 40, 1)
        dtype = "uint16"
        test_image = [np.zeros(dims[:2], dtype=dtype)] * dims[2]
        mosaic_rows = 2
        mosaic_cols = 3
        exposure_times = [2., 3., 4., 5., 0., 0.]
        # 0 exposures should be ignored, if they exist, as these will be interrupted frames
        expected_exposure = 3.5

        ole = MagicMock()
        ole.exists.return_value = True
        mocked_extractor.extract_multiple_exposure_times.return_value = exposure_times
        mocked_extractor.extract_pixel_size.return_value = 0.005
        mocked_extractor.extract_x_coords.return_value = [7.5, 22.5, 37.5] * 2
        mocked_extractor.extract_y_coords.return_value = [10, 10, 10, 20, 20, 20]
        mocked_extractor.read_imageinfo_as_int.side_effect = [mosaic_rows, mosaic_cols]

        ome_metadata = create_ome_metadata(ole, test_image)
        self.assertEqual(ome_metadata.image().Pixels.Plane(0).get_ExposureTime(), expected_exposure)

    @patch('txrm2tiff.txrm_to_image.txrm_wrapper')
    def test_mosaic_centre_found_correctly(self, mocked_extractor):
        dims = (45, 40, 1)
        dtype = "uint16"
        test_image = [np.zeros(dims[:2], dtype=dtype)] * dims[2]
        pixel_size = 0.005
        mosaic_cols = 3
        mosaic_rows = 2
        exposure_times = [0., 0., 2., 3., 4., 5.]
        offset = [3., -2.]
        # Multiplier of 1.e3 required as units from xrm files are micrometres and the output should be in nanometres
        expected_centre = ((22.5 + offset[0]) * pixel_size * 1.e3, (20. + offset[1]) * pixel_size * 1.e3)

        
        image_divider = MagicMock()
        txrm_converter = TxrmToImage()
        ole = MagicMock()
        ole.exists.return_value = True
        mocked_extractor.extract_multiple_exposure_times.return_value = exposure_times
        mocked_extractor.extract_pixel_size.return_value = pixel_size

        # This should only need the coords of the first frame as mosaic may not complete
        mocked_extractor.extract_x_coords.return_value = [
            (7.5 + offset[0]) * pixel_size, 0, 0, 0, 0, 0]
        mocked_extractor.extract_y_coords.return_value = [
            (10 + offset[1]) * pixel_size, 0, 0, 0, 0, 0]

        mocked_extractor.read_imageinfo_as_int.side_effect = [mosaic_rows, mosaic_cols]

        ome_metadata = create_ome_metadata(ole, test_image)
        plane = ome_metadata.image().Pixels.Plane(0)
        ome_centre = (float(plane.get_PositionX()), float(plane.get_PositionY()))
        [self.assertAlmostEqual(ome, expected) for ome, expected in zip(ome_centre, expected_centre)]

    def test_save_before_convert(self):
        with self.assertRaises(IOError):
            TxrmToImage().save("output_path")

    @patch('txrm2tiff.txrm_to_image.isOleFile', MagicMock(return_value=True))
    @patch('txrm2tiff.txrm_to_image.OleFileIO')
    @patch('txrm2tiff.txrm_wrapper.extract_all_images')
    def test_get_reference_custom_median(self, mocked_extractor, mocked_olefile):
        custom_reference = []
        ole = MagicMock()
        for i in range(1, 5):
            custom_reference.append(np.full((5,5), i))
        custom_reference.append(np.full((5,5), 7))
        mocked_extractor.return_value = custom_reference
        ref_ole = MagicMock()
        mocked_olefile.return_value = ref_ole
        ref = _get_reference(ole, "txrm_name", ref_ole, ignore_reference=False)
        assert_array_equal(ref, custom_reference[2])

    @patch('txrm2tiff.txrm_to_image.isOleFile', MagicMock(return_value=True))
    @patch('txrm2tiff.txrm_to_image.OleFileIO')
    @patch('txrm2tiff.txrm_wrapper.extract_all_images') 
    def test_get_reference_custom_flat(self, mocked_extractor, mocked_olefile):
        ole = MagicMock()
        custom_reference = [np.full((5,5), 4)]
        mocked_extractor.return_value = custom_reference
        ref_ole = MagicMock()
        mocked_olefile.return_value = ref_ole
        ref = _get_reference(ole, "txrm_name", ref_ole, ignore_reference=False)
        assert_array_equal(ref, custom_reference[0])


base_path = Path("/dls/science/groups/das/ExampleData/B24_test_data/data/2019/cm98765-1")
raw_path = base_path / "raw"
xm10_path = raw_path / "XMv10"
xm13_path = raw_path / "XMv13"

test_files = [
    (xm13_path / 'Xray_mosaic_v13.xrm', ),
    (xm13_path / 'Xray_mosaic_v13_interrupt.xrm', ),
    (xm13_path / 'Xray_single_v13.xrm', ),
    (xm13_path / 'tomo_v13_full.txrm', ),
    (xm13_path / 'tomo_v13_full_noref.txrm', ),
    (xm13_path / 'tomo_v13_interrupt.txrm', ),
    (xm13_path / 'VLM_mosaic_v13.xrm', ),
    (xm13_path / 'VLM_mosaic_v13_interrupt.xrm', ),
    (xm10_path / '12_Tomo_F4D_Area1_noref.txrm', ),
    (xm10_path / 'VLM_mosaic.xrm', ),
    (xm10_path / 'test_tomo2_e3C_full.txrm', ),
    (xm10_path / 'Xray_mosaic_F5A.xrm', ),
]


@unittest.skipIf(not base_path.exists(), "dls paths cannot be accessed")
class TestTxrmToImageWithFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_path = base_path
        cls.raw_path = raw_path
        cls.processed_path = cls.base_path / "processed"
        if cls.processed_path.exists():
            cleanup_timeout = 40
            # rmtree is asynchronous, so a wait may be required:
            rmtree(cls.processed_path, ignore_errors=True)
            start = time()
            while os.path.exists(cls.processed_path) and (time() - start) < cleanup_timeout:
                sleep(1)

    def setUp(self):
        self.processed_path.mkdir()
        self.assertTrue(self.processed_path.exists(), msg="Processed folder not correctly created")
    
    def tearDown(self):
        cleanup_timeout = 40
        # rmtree is asynchronous, so a wait may be required:
        rmtree(self.processed_path, ignore_errors=True)
        start = time()
        while os.path.exists(self.processed_path) and (time() - start) < cleanup_timeout:
            sleep(1)

        self.assertFalse(self.processed_path.exists(), msg="Processed folder not correctly removed")

    @parameterized.expand(test_files)
    def test_converts_to_tiff(self, test_file):
        logging.debug("Running with file %s", test_file)
        output_path = self.processed_path / test_file.relative_to(self.raw_path).with_suffix('.ome.tiff')
        converter = TxrmToImage()
        # Make processed/ subfolders:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        converter.convert(test_file, None, False)
        converter.save(output_path)

        self.assertTrue(output_path.exists())


suite1 = unittest.TestLoader().loadTestsFromTestCase(TestTxrmToImageSimple)
suite2 = unittest.TestLoader().loadTestsFromTestCase(TestTxrmToImageWithFiles)
alltests = unittest.TestSuite([suite1, suite2])

if __name__ == "__main__":
    unittest.TextTestRunner().run(alltests)