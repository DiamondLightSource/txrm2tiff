from datetime import datetime
import unittest
from unittest.mock import MagicMock

from txrm2tiff.utils.metadata import create_ome_metadata


class TestMetadata(unittest.TestCase):
    def test_metadata_created_correctly(self):
        filename = "test_file.ext"
        dims = (45, 40, 1)
        mosaic_rows = 2
        mosaic_cols = 3

        txrm = MagicMock()

        txrm.output_shape = dims[::-1]
        exposures = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        txrm.exposures = exposures
        txrm.image_info = {
            "PixelSize": [0.005],
            "XPosition": [7.5, 22.5, 37.5] * 2,
            "YPosition": [10, 10, 10, 20, 20, 20],
            "ZPosition": [1, 1, 1, 1, 1, 1],
        }
        txrm.datetimes = [datetime(2021, 12, 30, 23, 55, 59)]
        txrm.mosaic_dims = [mosaic_cols, mosaic_rows]
        txrm.is_mosaic = False

        ome_metadata = create_ome_metadata(txrm, filename)
        self.assertEqual(ome_metadata.image().Pixels.get_SizeX(), dims[0])
        self.assertEqual(ome_metadata.image().Pixels.get_SizeY(), dims[1])
        self.assertEqual(ome_metadata.image().Pixels.get_SizeT(), dims[2])
        self.assertEqual(
            ome_metadata.image().get_AcquisitionDate(), "2021-12-30T23:55:59"
        )

    def test_mosaic_exposure_averaged(self):
        filename = "test_file.ext"
        dims = (45, 40, 1)
        mosaic_rows = 2
        mosaic_cols = 3

        txrm = MagicMock()

        txrm.output_shape = dims[::-1]
        exposures = [
            2.0,
            3.0,
            4.0,
            5.0,
            0.0,
            0.0,
        ]  # 0 exposures should be ignored, if they exist, as these will be interrupted frames
        txrm.exposures = exposures
        txrm.image_info = {
            "PixelSize": [0.005],
            "XPosition": [7.5, 22.5, 37.5] * 2,
            "YPosition": [10, 10, 10, 20, 20, 20],
            "ZPosition": [1, 1, 1, 1, 1, 1],
        }
        txrm.datetimes = [datetime(2021, 12, 30, 23, 55, 59)]
        txrm.mosaic_dims = [mosaic_cols, mosaic_rows]
        txrm.is_mosaic = True

        expected_exposure = 3.5

        ome_metadata = create_ome_metadata(txrm, filename)
        self.assertEqual(
            ome_metadata.image().Pixels.Plane(0).get_ExposureTime(), expected_exposure
        )

    def test_mosaic_centre_found_correctly(self):
        filename = "test_file.ext"
        dims = (45, 40, 1)
        pixel_size = 0.005
        mosaic_cols = 3
        mosaic_rows = 2
        exposure_times = [0.0, 0.0, 2.0, 3.0, 4.0, 5.0]
        offset = [3.0, -2.0]
        # Multiplier of 1.e3 required as units from xrm files are micrometres and the output should be in nanometres
        expected_centre = (
            (22.5 + offset[0]) * pixel_size * 1.0e3,
            (20.0 + offset[1]) * pixel_size * 1.0e3,
        )

        txrm = MagicMock()

        txrm.output_shape = dims[::-1]
        txrm.exposures = exposure_times
        # This should only need the coords of the first frame as mosaic may not complete
        txrm.image_info = {
            "PixelSize": [pixel_size],
            "XPosition": [
                (7.5 + offset[0]) * pixel_size,
                0,
                0,
                0,
                0,
                0,
            ],
            "YPosition": [
                (10 + offset[1]) * pixel_size,
                0,
                0,
                0,
                0,
                0,
            ],
            "ZPosition": [1, 1, 1, 1, 1, 1],
        }
        txrm.datetimes = [datetime(2021, 12, 30, 23, 55, 59)]
        txrm.mosaic_dims = [mosaic_cols, mosaic_rows]
        txrm.is_mosaic = True

        ome_metadata = create_ome_metadata(txrm, filename)
        plane = ome_metadata.image().Pixels.Plane(0)
        ome_centre = (float(plane.get_PositionX()), float(plane.get_PositionY()))
        [
            self.assertAlmostEqual(ome, expected)
            for ome, expected in zip(ome_centre, expected_centre)
        ]
