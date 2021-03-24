import unittest
from unittest.mock import MagicMock, patch
from numpy.testing import assert_array_equal, assert_array_almost_equal
from datetime import datetime
from random import randint

from parameterized import parameterized

import numpy as np
import os
import logging
import sys
import tifffile as tf
from pathlib import Path
from functools import partial
from shutil import rmtree
from time import time, sleep
from olefile import OleFileIO

from txrm2tiff.txrm_to_image import TxrmToImage, _get_reference, _apply_reference, create_ome_metadata, _conditional_replace, _stitch_images


class TestTxrmToImageSimple(unittest.TestCase):

    def test_divides_images_by_reference(self):
        num_images = 5
        images = [np.array([[0, 2, 4], [6, 8, 10]])] * num_images
        reference = np.arange(6).reshape(2, 3)
        resultant_images = _apply_reference(images, reference)
        self.assertEqual(len(resultant_images), num_images, msg="The result is the wrong length")
        expected_image = np.array([[0, 200, 200], [200, 200, 200]]).astype(resultant_images[0].dtype)
        for image in resultant_images:
            assert_array_equal(image, expected_image, err_msg="The result does not match the expected result")
 
    def test_referenced_image_is_float32(self):
        expected_dtype = np.float32
        num_images = 5
        images = [np.array([[0, 2, 4], [6, 8, 10]])] * num_images
        reference = np.arange(6).reshape(2, 3)
        resultant_images = _apply_reference(images, reference)
        self.assertEqual(len(resultant_images), num_images, msg="The result is the wrong length")
        for image in resultant_images:
            self.assertEqual(image.dtype, expected_dtype, msg=f"The dtype is {image.dtype} not {expected_dtype}")


    def test_conditional_replace(self):
        array = np.repeat(np.arange(0, 9000, 0.1).reshape(300,300), 10, 0)
        threshold = 100

        _conditional_replace(array, np.nan, lambda x: x < threshold)

        self.assertEqual(
            np.nanmin(array), threshold,
            msg="Array values below threshold {} have not been replaced with nan".format(threshold))


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

    def test_stitch_mosaic(self):
        mosaic_xy_shape = (3, 4)
        image_size = (400, 400)
        fast_axis = 1
        slow_axis = 1 - fast_axis
        images = np.zeros((mosaic_xy_shape[slow_axis] * mosaic_xy_shape[fast_axis], image_size[fast_axis], image_size[slow_axis]))
        range_array = np.repeat(np.arange(0, mosaic_xy_shape[fast_axis]), image_size[0] *  image_size[1])
        images.flat = list(range_array) * mosaic_xy_shape[slow_axis]

        output_image = _stitch_images(images, mosaic_xy_shape, fast_axis)

        expected_array = np.concatenate(
            [np.concatenate(
                [np.full(image_size, i) for i in range(0, mosaic_xy_shape[fast_axis])],
                axis=fast_axis) for j in range(0, mosaic_xy_shape[slow_axis])],
                axis=slow_axis)[np.newaxis, :, :]
        
        assert_array_equal(output_image, expected_array)

    def test_save_before_convert(self):
        with self.assertRaises(IOError):
            TxrmToImage().save("output_path")

    @patch('txrm2tiff.txrm_to_image.isOleFile', MagicMock(return_value=True))
    @patch('txrm2tiff.txrm_to_image.OleFileIO')
    @patch('txrm2tiff.txrm_wrapper.extract_all_images')
    def test_get_reference_custom_despeckle_ave(self, mocked_extractor, mocked_olefile):
        custom_reference = []
        ole = MagicMock()
        dims = (250, 250)
        speckle_per_frame = 5
        mid_point = 10
        for i in range(0, mid_point * 2 + 1):
            speckle_array = np.full(dims, i)
            for _ in range(speckle_per_frame):
                speckle_idx = (randint(0, dims[0] - 1), randint(0, dims[1] - 1))
                speckle_array[speckle_idx] = i * randint(500, 1000)  # Should be well beyond 2.8 standard devs
            custom_reference.append(speckle_array)
        mocked_extractor.return_value = np.asarray(custom_reference)
        ref_ole = MagicMock()
        mocked_olefile.return_value = ref_ole
        ref = _get_reference(ole, "txrm_name", Path("ref/path.txrm"), ignore_reference=False)
        assert_array_almost_equal(ref, np.full(dims, mid_point, dtype=np.float32), decimal=0)

    @patch('txrm2tiff.txrm_to_image.isOleFile', MagicMock(return_value=True))
    @patch('txrm2tiff.txrm_to_image.OleFileIO')
    @patch('txrm2tiff.txrm_wrapper.extract_all_images')
    def test_get_reference_custom_flat(self, mocked_extractor, mocked_olefile):
        ole = MagicMock()
        custom_reference = [np.full((5,5), 4)]
        mocked_extractor.return_value = custom_reference
        ref_ole = MagicMock()
        mocked_olefile.return_value = ref_ole
        ref = _get_reference(ole, "txrm_name", Path("ref/path.txrm"), ignore_reference=False)
        assert_array_equal(ref, custom_reference[0])


visit_path = Path("/dls/science/groups/das/ExampleData/B24_test_data/data/2019/cm98765-1")
raw_path = visit_path / "raw"
xm10_path = raw_path / "XMv10"
xm13_path = raw_path / "XMv13"

test_files = [
    (xm13_path / 'Xray_mosaic_v13.xrm', ),
    (xm13_path / 'Xray_mosaic_v13_interrupt.xrm', ),
    (xm13_path / 'Xray_mosaic_7x7_v13.xrm', ),
    (xm13_path / 'Xray_single_v13.xrm', ),
    (xm13_path / 'tomo_v13_full.txrm', ),
    (xm13_path / 'tomo_v13_full_noref.txrm', ),
    (xm13_path / 'tomo_v13_interrupt.txrm', ),
    (xm13_path / 'VLM_mosaic_v13.xrm', ),
    (xm13_path / 'VLM_mosaic_v13_interrupt.xrm', ),
    (xm13_path / 'VLM_grid_mosaic_large_v13.xrm', ),
    (xm10_path / '12_Tomo_F4D_Area1_noref.txrm', ),
    (xm10_path / 'VLM_mosaic.xrm', ),
    (xm10_path / 'test_tomo2_e3C_full.txrm', ),
    (xm10_path / 'Xray_mosaic_F5A.xrm', ),
    ]

@unittest.skipUnless(visit_path.exists(), "dls paths cannot be accessed")
class TestTxrmToImageWithFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.visit_path = visit_path
        cls.raw_path = raw_path
        cls.processed_path = cls.visit_path / "processed"
        if cls.processed_path.exists():
            cleanup_timeout = 40
            # rmtree is asynchronous, so a wait may be required:
            rmtree(cls.processed_path, ignore_errors=True)
            start = time()
            while os.path.exists(cls.processed_path) and (time() - start) < cleanup_timeout:
                sleep(1)

    def setUp(self):
        self.processed_path.mkdir(exist_ok=True)
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


    def test_converts_to_tiff_with_dtype(self):
        test_file = test_files[0][0]
        dtypes = ['uint16', 'float32', 'float64', np.float32, np.float64, np.uint16]
        logging.debug("Running with file %s", test_file)
        converter = TxrmToImage()

        converter.convert(test_file, None, False)
        for dtype in dtypes:
            output_path = self.processed_path / (test_file.parent / f"{test_file.stem}_{dtype}.ome.tiff").relative_to(self.raw_path)
            # Make processed/ subfolders:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            converter.save(output_path, data_type=dtype)
            self.assertTrue(output_path.exists())
            with tf.TiffFile(str(output_path)) as tif:
                a = tif.asarray()
            self.assertEqual(a.dtype, np.dtype(dtype), msg=f"dtype is {a.dtype} not {dtype}")

        self.assertTrue(output_path.exists())


suite1 = unittest.TestLoader().loadTestsFromTestCase(TestTxrmToImageSimple)
suite2 = unittest.TestLoader().loadTestsFromTestCase(TestTxrmToImageWithFiles)
alltests = unittest.TestSuite([suite1, suite2])

if __name__ == "__main__":
    unittest.TextTestRunner().run(alltests)
