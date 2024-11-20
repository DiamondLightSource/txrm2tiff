import unittest
from parameterized import parameterized

import logging
import os
from pathlib import Path
from time import time, sleep
from shutil import rmtree
from io import IOBase
import numpy as np
import tifffile as tf

import txrm2tiff


visit_path = Path(
    "/dls/science/groups/das/ExampleData/B24_test_data/data/2019/cm98765-1"
)
raw_path = visit_path / "raw"
xm10_path = raw_path / "XMv10"
xm13_path = raw_path / "XMv13"

test_files = [
    (xm13_path / "Xray_mosaic_v13.xrm",),
    (xm13_path / "Xray_mosaic_v13_interrupt.xrm",),
    (xm13_path / "Xray_mosaic_7x7_v13.xrm",),
    (xm13_path / "Xray_single_v13.xrm",),
    (xm13_path / "tomo_v13_full.txrm",),
    (xm13_path / "tomo_v13_full_noref.txrm",),
    (xm13_path / "tomo_v13_interrupt.txrm",),
    (xm13_path / "VLM_mosaic_v13.xrm",),
    (xm13_path / "VLM_mosaic_v13_interrupt.xrm",),
    (xm13_path / "VLM_grid_mosaic_large_v13.xrm",),
    (xm10_path / "12_Tomo_F4D_Area1_noref.txrm",),
    (xm10_path / "VLM_mosaic.xrm",),
    (xm10_path / "test_tomo2_e3C_full.txrm",),
    (xm10_path / "Xray_mosaic_F5A.xrm",),
]


@unittest.skipUnless(visit_path.exists(), "dls paths cannot be accessed")
class TestTxrm2TiffWithFiles(unittest.TestCase):
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
            while (
                os.path.exists(cls.processed_path)
                and (time() - start) < cleanup_timeout
            ):
                sleep(1)

    def setUp(self):
        self.processed_path.mkdir(exist_ok=True)
        self.assertTrue(
            self.processed_path.exists(), msg="Processed folder not correctly created"
        )

    def tearDown(self):
        cleanup_timeout = 40
        # rmtree is asynchronous, so a wait may be required:
        rmtree(self.processed_path, ignore_errors=True)
        start = time()
        while (
            os.path.exists(self.processed_path) and (time() - start) < cleanup_timeout
        ):
            sleep(1)

        self.assertFalse(
            self.processed_path.exists(), msg="Processed folder not correctly removed"
        )

    @parameterized.expand(test_files)
    def test_convert_and_save(self, test_file):
        logging.debug("Running with file %s", test_file)
        output_path = self.processed_path / test_file.relative_to(
            self.raw_path
        ).with_suffix(".ome.tiff")

        # Make processed/ subfolders:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        txrm2tiff.convert_and_save(test_file, output_path=output_path)

        self.assertTrue(output_path.is_file())

    def test_convert_and_save_with_dtype(self):
        test_file = test_files[0][0]
        dtypes = ["uint16", "float32", "float64", np.float32, np.float64, np.uint16]
        logging.debug("Running with file %s", test_file)

        for dtype in dtypes:
            output_path = self.processed_path / (
                test_file.parent / f"{test_file.stem}_{dtype}.ome.tiff"
            ).relative_to(self.raw_path)
            # Make processed/ subfolders:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            txrm2tiff.convert_and_save(
                test_file, output_path=output_path, data_type=dtype
            )

            self.assertTrue(output_path.exists())
            with tf.TiffFile(str(output_path)) as tif:
                a = tif.asarray()
            self.assertEqual(
                a.dtype, np.dtype(dtype), msg=f"dtype is {a.dtype} not {dtype}"
            )

    @parameterized.expand(test_files)
    def test_convert_with_txrm_class(self, test_file):
        logging.debug("Running with file %s", test_file)
        output_path = self.processed_path / test_file.relative_to(
            self.raw_path
        ).with_suffix(".ome.tiff")

        # Make processed/ subfolders:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with txrm2tiff.open_txrm(test_file, strict=True) as txrm:
            txrm.apply_reference()
            self.assertTrue(txrm.save_images(output_path))

        self.assertTrue(output_path.is_file())

    def test_convert_with_txrm_class_with_dtype(self):
        test_file = test_files[0][0]
        dtypes = ["uint16", "float32", "float64", np.float32, np.float64, np.uint16]
        logging.debug("Running with file %s", test_file)

        with txrm2tiff.open_txrm(test_file, strict=True) as txrm:
            if txrm.has_reference:
                txrm.apply_reference()
            for dtype in dtypes:
                output_path = self.processed_path / (
                    test_file.parent / f"{test_file.stem}_{dtype}.ome.tiff"
                ).relative_to(self.raw_path)
                # Make processed/ subfolders:
                output_path.parent.mkdir(parents=True, exist_ok=True)

                self.assertTrue(txrm.save_images(output_path, dtype))

                self.assertTrue(output_path.exists())
                with tf.TiffFile(str(output_path)) as tif:
                    a = tif.asarray()
                self.assertEqual(
                    a.dtype, np.dtype(dtype), msg=f"dtype is {a.dtype} not {dtype}"
                )

    def test_convert_with_txrm_class_using_open_file_buffer(self):
        test_file = test_files[0][0]
        logging.debug("Running with file %s", test_file)

        with test_file.open("rb") as f:
            self.assertTrue(isinstance(f, IOBase))

            with txrm2tiff.open_txrm(f, strict=True) as txrm:
                if txrm.has_reference:
                    txrm.apply_reference()
                output_path = self.processed_path / (
                    test_file.parent / f"{test_file.stem}.ome.tiff"
                ).relative_to(self.raw_path)
                # Make processed/ subfolders:
                output_path.parent.mkdir(parents=True, exist_ok=True)

                self.assertTrue(txrm.save_images(output_path))

        self.assertTrue(output_path.is_file())

    def test_convert_with_txrm_class_using_bytes(self):
        test_file = test_files[0][0]
        logging.debug("Running with file %s", test_file)

        with test_file.open("rb") as f:
            self.assertTrue(isinstance(f, IOBase))
            bytestring = f.read()

        with txrm2tiff.open_txrm(bytestring, strict=True) as txrm:
            if txrm.has_reference:
                txrm.apply_reference()
            output_path = self.processed_path / (
                test_file.parent / f"{test_file.stem}.ome.tiff"
            ).relative_to(self.raw_path)
            # Make processed/ subfolders:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.assertTrue(txrm.save_images(output_path))

        self.assertTrue(output_path.is_file())
