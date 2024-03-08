from datetime import datetime
import unittest
from unittest.mock import MagicMock, patch

from txrm2tiff.txrm import meta_mixin


class TestMetadata(unittest.TestCase):
    def test_metadata_created_correctly(self):
        filename = "test_file.ext"
        dims = (45, 40, 1)
        mosaic_rows = 2
        mosaic_cols = 3
        
        with patch.object(meta_mixin.model, "Instrument") as mocked_instrument:
            mocked_instrument.return_value = None
            txrm = meta_mixin.MetaMixin()
            txrm.strict = False
            txrm.file_is_open = False
            txrm.name = filename
            txrm.output_shape = dims[::-1]
            exposures = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            txrm.exposures = exposures
            txrm.image_info = {
                "PixelSize": [0.005],  # microns
                "XPosition": [7.5, 22.5, 37.5] * 2,
                "YPosition": [10, 10, 10, 20, 20, 20],
                "ZPosition": [1, 1, 1, 1, 1, 1],
            }
            
            txrm.energies = [500.] * len(exposures)
            acquisition_time = datetime(2021, 12, 30, 23, 55, 59)
            txrm.datetimes = [acquisition_time]
            txrm.mosaic_dims = [mosaic_cols, mosaic_rows]
            txrm.is_mosaic = False

            pixel_size_nm = txrm.image_info["PixelSize"][0] * 1.e3
            
            print(txrm.metadata)
            ome_metadata = txrm.metadata
            self.assertEqual(ome_metadata.instruments, [])
            self.assertIsNone(ome_metadata.images[0].instrument_ref)
            self.assertEqual(ome_metadata.images[0].pixels.physical_size_x, pixel_size_nm)
            self.assertEqual(ome_metadata.images[0].pixels.physical_size_y, pixel_size_nm)
            self.assertEqual(ome_metadata.images[0].pixels.physical_size_z, 1)
            self.assertEqual(ome_metadata.images[0].pixels.size_x, dims[0])
            self.assertEqual(ome_metadata.images[0].pixels.size_y, dims[1])
            self.assertEqual(ome_metadata.images[0].pixels.size_z, dims[2])
            self.assertEqual(
                ome_metadata.images[0].acquisition_date, acquisition_time
            )
            

    def test_mosaic_exposure_averaged(self):
        filename = "test_file.ext"
        dims = (45, 40, 1)
        mosaic_rows = 2
        mosaic_cols = 3

        with patch.object(meta_mixin.model, "Instrument") as mocked_instrument:
            txrm = meta_mixin.MetaMixin()
            txrm.strict = False
            txrm.file_is_open = False

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
            txrm.energies = [500.] * len(exposures)

            expected_exposure = 3.5

            self.assertEqual(
                txrm.metadata.images[0].pixels.planes[0].exposure_time, expected_exposure
            )

    def test_mosaic_centre_found_correctly(self):
        filename = "test_file.ext"
        dims = (45, 40, 1)
        pixel_size = 0.005
        mosaic_cols = 3
        mosaic_rows = 2
        exposures = [0.0, 0.0, 2.0, 3.0, 4.0, 5.0]
        offset = [3.0, -2.0]
        # Multiplier of 1.e3 required as units from xrm files are micrometres and the output should be in nanometres
        expected_centre = (
            (22.5 + offset[0]) * pixel_size * 1.0e3,
            (20.0 + offset[1]) * pixel_size * 1.0e3,
        )

        with patch.object(meta_mixin.model, "Instrument") as mocked_instrument:
            txrm = meta_mixin.MetaMixin()
            txrm.strict = False
            txrm.file_is_open = False

            txrm.output_shape = dims[::-1]
            txrm.exposures = exposures
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
            txrm.energies = [500.] * len(exposures)

            ome_metadata = txrm.metadata
            plane = ome_metadata.images[0].pixels.planes[0]
            ome_centre = (float(plane.position_x), float(plane.position_y))
            [
                self.assertAlmostEqual(ome, expected)
                for ome, expected in zip(ome_centre, expected_centre)
            ]
