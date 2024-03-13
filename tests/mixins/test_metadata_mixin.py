import unittest
from unittest.mock import MagicMock, patch, call
from parameterized import parameterized

from datetime import datetime

from ome_types import model
from txrm2tiff.txrm import meta_mixin
from txrm2tiff.xradia_properties.enums import (
    XrmDataTypes,
    XrmSourceType,
)


class TestMetadataMixin(unittest.TestCase):
    def test_metadata_created_correctly(self):
        filename = "test_file.ext"
        dims = (45, 40, 1)
        mosaic_rows = 2
        mosaic_cols = 3

        with patch.object(meta_mixin.MetaMixin, "_ome_instrument", None):
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
                "Binning": [],
            }

            txrm.energies = [500.0] * len(exposures)
            acquisition_time = datetime(2021, 12, 30, 23, 55, 59)
            txrm.datetimes = [acquisition_time]
            txrm.mosaic_dims = [mosaic_cols, mosaic_rows]
            txrm.is_mosaic = False

            pixel_size_nm = txrm.image_info["PixelSize"][0] * 1.0e3

            ome_metadata = txrm.metadata
            self.assertEqual(ome_metadata.instruments, [])
            self.assertIsNone(ome_metadata.images[0].instrument_ref)
            self.assertEqual(
                ome_metadata.images[0].pixels.physical_size_x, pixel_size_nm
            )
            self.assertEqual(
                ome_metadata.images[0].pixels.physical_size_y, pixel_size_nm
            )
            self.assertEqual(ome_metadata.images[0].pixels.physical_size_z, 1)
            self.assertEqual(ome_metadata.images[0].pixels.size_x, dims[0])
            self.assertEqual(ome_metadata.images[0].pixels.size_y, dims[1])
            self.assertEqual(ome_metadata.images[0].pixels.size_z, dims[2])
            self.assertEqual(ome_metadata.images[0].acquisition_date, acquisition_time)

    def test_mosaic_exposure_averaged(self):
        filename = "test_file.ext"
        dims = (45, 40, 1)
        mosaic_rows = 2
        mosaic_cols = 3

        with patch.object(meta_mixin.MetaMixin, "_ome_instrument", None):
            txrm = meta_mixin.MetaMixin()
            txrm.name = filename
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
            txrm.energies = [500.0] * len(exposures)

            expected_exposure = 3.5

            self.assertEqual(
                txrm.metadata.images[0].pixels.planes[0].exposure_time,
                expected_exposure,
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

        with patch.object(meta_mixin.MetaMixin, "_ome_instrument", None):
            txrm = meta_mixin.MetaMixin()
            txrm.name = filename
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
            txrm.energies = [500.0] * len(exposures)

            ome_metadata = txrm.metadata
            plane = ome_metadata.images[0].pixels.planes[0]
            ome_centre = (float(plane.position_x), float(plane.position_y))
            [
                self.assertAlmostEqual(ome, expected)
                for ome, expected in zip(ome_centre, expected_centre)
            ]

    def test_ome_configured_camera_count(self):
        camera_count = 5
        txrm = meta_mixin.MetaMixin()
        txrm.read_stream = MagicMock(return_value=[camera_count])
        self.assertEqual(txrm._ome_configured_camera_count, camera_count)
        txrm.read_stream.assert_called_once_with(
            "ConfigureBackup/ConfigCamera/NumberOfCamera",
            XrmDataTypes.XRM_UNSIGNED_INT,
        )

    def test_ome_configured_detectors(self):
        camera_count = 5
        camera_ids = [f"camera {i}" for i in range(camera_count)]
        detectors = [f"detector {i}" for i in range(camera_count)]
        with (
            patch.object(meta_mixin.MetaMixin, "_camera_ids", camera_ids),
            patch.object(
                meta_mixin.MetaMixin,
                "_get_detector",
                MagicMock(side_effect=lambda i: f"detector {i}"),
            ),
            patch.object(
                meta_mixin.MetaMixin, "_ome_configured_camera_count", camera_count
            ),
        ):
            txrm = meta_mixin.MetaMixin()
            self.assertEqual(
                txrm._ome_configured_detectors,
                {cam: det for cam, det in zip(camera_ids, detectors)},
            )
            txrm._get_detector.assert_has_calls([call(i) for i in range(camera_count)])

    def test_get_detector(self):
        index = 999
        preamp_gain = 2.5
        detector_model = "model name"
        output_gain = 5.9
        txrm = meta_mixin.MetaMixin()
        txrm.read_stream = MagicMock(
            side_effect=[[preamp_gain], [output_gain], [detector_model]]
        )

        detector = txrm._get_detector(index)
        self.assertEqual(
            detector,
            model.Detector(
                id=f"Detector:{index}",
                gain=preamp_gain,
                amplification_gain=output_gain,
                model=detector_model,
            ),
        )
        stream_stem = f"ConfigureBackup/ConfigCamera/Camera {index + 1}"
        txrm.read_stream.assert_has_calls(
            [
                call(f"{stream_stem}/PreAmpGain", XrmDataTypes.XRM_FLOAT),
                call(f"{stream_stem}/OutputAmplifier", XrmDataTypes.XRM_FLOAT),
                call(f"{stream_stem}/CameraName", XrmDataTypes.XRM_STRING),
            ]
        )

    def test_camera_ids(self):
        camera_count = 5
        camera_indexes = tuple(range(camera_count))
        with patch.object(
            meta_mixin.MetaMixin, "_ome_configured_camera_count", camera_count
        ):
            txrm = meta_mixin.MetaMixin()
            txrm._get_camera_id = MagicMock(side_effect=lambda i: f"camera {i}")

            self.assertEqual(txrm._camera_ids, [f"camera {i}" for i in camera_indexes]),
        txrm._get_camera_id.assert_has_calls([call(i) for i in camera_indexes])

    def test_get_camera_id(self):
        index = 999
        camera_id = "camera id"
        txrm = meta_mixin.MetaMixin()
        txrm.read_stream = MagicMock(side_effect=[[camera_id]])
        self.assertEqual(txrm._get_camera_id(index), camera_id)
        txrm.read_stream.assert_called_once_with(
            f"ConfigureBackup/ConfigCamera/Camera {index + 1}/CameraID",
            XrmDataTypes.XRM_UNSIGNED_INT,
        )

    def test_ome_detector(self):
        camera_number = 3
        detector = "detector"
        with patch.object(
            meta_mixin.MetaMixin, "_ome_configured_detectors", {camera_number: detector}
        ):
            txrm = meta_mixin.MetaMixin()
            txrm.image_info = {"CameraNo": [camera_number]}
            self.assertEqual(txrm._ome_detector, detector)

    @parameterized.expand([("has ID", "Machine ID"), ("no ID", None)])
    def test_ome_microscope(self, _name, machine_id):
        txrm = meta_mixin.MetaMixin()
        has_machine_id = machine_id is not None
        txrm.has_stream = MagicMock(return_value=has_machine_id)
        kwargs = {}
        if has_machine_id:
            txrm.read_stream = MagicMock(return_value=[machine_id])
            kwargs["model"] = machine_id
        microscope = txrm._ome_microscope
        self.assertEqual(
            microscope,
            model.Microscope(
                type=model.Microscope_Type.OTHER, manufacturer="Xradia", **kwargs
            ),
            msg=f"Returned {microscope}",
        )
        txrm.has_stream.assert_called_once_with(
            "ConfigureBackup/XRMConfiguration/MachineID"
        )

    def test_ome_configured_objectives(self):
        camera_count = 5
        obj_count = 3
        camera_ids = [f"camera {i}" for i in range(camera_count)]
        objectives = [
            {f"obj {i} {j}": f"objective {i} {j}" for i in range(obj_count)}
            for j in range(camera_count)
        ]
        with (
            patch.object(meta_mixin.MetaMixin, "_camera_ids", camera_ids),
            patch.object(
                meta_mixin.MetaMixin,
                "_get_objectives",
                MagicMock(side_effect=lambda i: objectives[i]),
            ),
            patch.object(
                meta_mixin.MetaMixin, "_ome_configured_camera_count", camera_count
            ),
        ):
            txrm = meta_mixin.MetaMixin()
            self.assertEqual(
                txrm._ome_configured_objectives,
                {cam: obj for cam, obj in zip(camera_ids, objectives)},
            )
            txrm._get_objectives.assert_has_calls(
                [call(i) for i in range(camera_count)]
            )

    @parameterized.expand(
        [
            ("zoneplate names", None),
            ("no zoneplate names", [f"zp {i}" for i in range(3)]),
        ]
    )
    def test_get_objectives(self, _name, zoneplate_names):
        index = 999
        objective_count = 3
        objective_names = [f"obj name {i}" for i in range(objective_count)]
        objective_ids = [f"obj name {i}" for i in range(objective_count)]
        magnifications = [0.5 for _ in range(objective_count)]
        txrm = meta_mixin.MetaMixin()
        txrm.read_stream = MagicMock(
            side_effect=[
                objective_ids,
                magnifications,
                objective_names,
                zoneplate_names,
            ]
        )

        if zoneplate_names is None:

            expected_objectives = {
                objective_name: model.Objective(
                    id=f"Objective:{obj_id}.0",
                    nominal_magnification=magnification,
                    model=objective_name,
                )
                for obj_id, objective_name, magnification in zip(
                    objective_ids, objective_names, magnifications
                )
            }
        else:
            expected_objectives = {
                f"{objective_name}_{zp_name}": model.Objective(
                    id=f"Objective:{obj_id}.{zp_number}",
                    nominal_magnification=magnification,
                    model=zp_name,
                )
                for zp_number, zp_name in enumerate(zoneplate_names)
                for obj_id, objective_name, magnification in zip(
                    objective_ids, objective_names, magnifications
                )
            }

        self.assertEqual(txrm._get_objectives(index), expected_objectives)

        stream_stem = f"ConfigureBackup/ConfigCamera/Camera {index + 1}"
        txrm.read_stream.assert_has_calls(
            [
                call(
                    f"{stream_stem}/ConfigObjectives/ObjectiveID",
                    XrmDataTypes.XRM_STRING,
                ),
                call(
                    f"{stream_stem}/ConfigObjectives/OpticalMagnification",
                    XrmDataTypes.XRM_FLOAT,
                ),
                call(
                    f"{stream_stem}/ConfigObjectives/ObjectiveName",
                    XrmDataTypes.XRM_STRING,
                ),
                call(
                    f"{stream_stem}/ConfigZonePlates/Name",
                    XrmDataTypes.XRM_STRING,
                ),
            ]
        )

    # TODO: Test whether the OME parsing works as intended
