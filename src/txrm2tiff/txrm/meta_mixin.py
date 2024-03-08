import logging
from scipy import constants
import numpy as  np
from ome_types import model
from ome_types.model.simple_types import UnitsLength, UnitsFrequency, UnitsTime, Binning
from ome_types.model.channel import AcquisitionMode, IlluminationType

from ..info import __version__
from .txrm_property import txrm_property
from ..xradia_properties.enums import (
    XrmDataTypes,
    XrmObjectiveType,
    XrmSourceType,
)
from ..utils.metadata import dtype_dict

def get_ome_pixel_type(dtype):
    try:
        return dtype_dict[np.dtype(dtype).name]
    except Exception:
        raise TypeError(f"{dtype} is unsupported data type")
        
class MetaMixin:
    @txrm_property(fallback=0)
    def _ome_configured_camera_count(self):
        return self.read_stream(
            "ConfigureBackup/ConfigCamera/NumberOfCamera", XrmDataTypes.XRM_UNSIGNED_INT
        )[0]

    @txrm_property(fallback=[])
    def _ome_configured_detectors(self):
        return [self._get_detector(i) for i in range(self._ome_configured_camera_count)]

    def _get_detector(self, index):
        stream_stem = f"ConfigureBackup/ConfigCamera/Camera {index + 1}"
        camera_name = self.read_stream(
            f"{stream_stem}/CameraName", XrmDataTypes.XRM_STRING
        )[0]
        return model.Detector(
            id=f"Detector:{index}",
            gain=self.read_stream(f"{stream_stem}/PreAmpGain", XrmDataTypes.XRM_FLOAT)[
                0
            ],
            amplification_gain=self.read_stream(
                f"{stream_stem}/OutputAmplifier", XrmDataTypes.XRM_FLOAT
            )[0],
            model=camera_name,
        )

    @txrm_property(fallback=None)
    def _ome_detector(self):
        camera_idx = (
            self.read_stream("ImageInfo/CameraNo", XrmDataTypes.XRM_UNSIGNED_INT)[0] - 1
        )  # Stream counts from 1
        return self._ome_configured_detectors[camera_idx]

    @txrm_property(fallback=None)
    def _ome_microscope(self):
        machine_id = self.read_stream(
            "ConfigureBackup/XRMConfiguration/MachineID", dtype=XrmDataTypes.XRM_STRING
        )
        kwargs = {}
        if machine_id:
            kwargs["model"] = machine_id[0]
        return model.Microscope(
            type="Other",
            manufacturer="Xradia",
            **kwargs,
        )

    @txrm_property(fallback=[])
    def _ome_configured_objectives(self):
        return [
            obj
            for i in range(self._ome_configured_camera_count)
            for obj in self._get_objectives(i)
        ]

    def _get_objectives(self, index):
        # TODO: Replace 'ObjectiveName' (a little reduntant) with zoneplate name, which is the actual optical objective
        # This is waiting on the zoneplate name actually being populated in the metadata.
        stream_stem = f"ConfigureBackup/ConfigCamera/Camera {index + 1}"
        name_stream = f"{stream_stem}/ConfigObjectives/ObjectiveName"
        objective_names = self.read_stream(name_stream, XrmDataTypes.XRM_STRING)
        objective_ids = [
            XrmObjectiveType(id_).name.replace(" ", "_")
            for id_ in self.read_stream(
                f"{stream_stem}/ConfigObjectives/ObjectiveID",
                XrmDataTypes.XRM_UNSIGNED_INT,
            )
        ]
        magnifications = self.read_stream(
            f"{stream_stem}/ConfigObjectives/OpticalMagnification",
            XrmDataTypes.XRM_FLOAT,
        )
        return [
            model.Objective(
                id=f"Objective:{id_}", nominal_magnification=magnification, model=name
            )
            for id_, name, magnification in zip(
                objective_ids, objective_names, magnifications
            )
        ]

    @txrm_property(fallback=None)
    def _ome_instrument(self):
        return model.Instrument(
            id="Instrument:0",
            detectors=self._ome_configured_detectors,
            microscope=self._ome_microscope,
            objectives=self._ome_configured_objectives,
            light_source_group=self._ome_configured_light_sources,
        )

    @txrm_property(fallback=None)
    def _ome_instrument_ref(self):
        if self._ome_instrument is None:
            return None
        return model.InstrumentRef(id=self._ome_instrument.id)

    @txrm_property(fallback=None)
    def _ome_objective(self):
        camera_idx = (
            self.read_stream("ImageInfo/CameraNo", XrmDataTypes.XRM_UNSIGNED_INT)[0] - 1
        )  # Stream counts from 1
        return self.configured_objectives[camera_idx]

    @txrm_property(fallback=None)
    def _ome_objective_ref(self):
        if self._ome_objective is None:
            return None
        return model.InstrumentRef(id=self._ome_objective.id)

    @txrm_property(fallback=0)
    def _ome_configured_source_count(self):
        return self.read_stream(
            "ConfigureBackup/ConfigSources/NumberOfSources",
            XrmDataTypes.XRM_UNSIGNED_INT,
        )[0]

    @txrm_property(fallback=[])
    def _ome_configured_light_sources(self):
        return [self._get_light_source(i) for i in range(self._ome_configured_source_count)]

    def _get_light_source(self, index):
        stream_stem = (
            f"ConfigureBackup/ConfigSources/Source {index + 1}"  # Stream from 1
        )
        source_type = XrmSourceType(
            self.read_stream(f"{stream_stem}/Type", XrmDataTypes.XRM_UNSIGNED_INT)[0]
        )
        id_ = f"LightSource:{index}"
        name = self.read_stream(f"{stream_stem}/SourceName", XrmDataTypes.XRM_STRING)[0]
        if source_type == XrmSourceType.XRM_VISUAL_LIGHT_SOURCE:
            source = model.LightEmittingDiode
        else:
            source = model.GenericExcitationSource

        return source(
            id=id_,
            model=name,
        )

    @txrm_property(fallback=None)
    def _ome_light_source(self):
        source_idx = self.read_stream(
            "AcquisitionSettings/SourceIndex", XrmDataTypes.XRM_UNSIGNED_INT
        )[
            0
        ]  # Stream counts from 0
        return self._ome_configured_light_sources[source_idx]

    @txrm_property(fallback=None)
    def _ome_light_source_settings(self):
        if self._ome_light_source is None:
            return None
        
        mean_energy = np.mean(self.energies)
        kwargs = {}
        if mean_energy:
            kwargs["wavelength"] = (
                1.0e9 * mean_energy / constants.electron_volt / constants.Planck
            )
            kwargs["wavelength_unit"] = UnitsLength.NANOMETER

        return model.LightSourceSettings(id=self._ome_light_source.id, **kwargs)

    @txrm_property(fallback=None)
    def _ome_detector_settings(self):
        return model.DetectorSettings(
            id=self.detector.id,
            binning=Binning(
                "{0}x{0}".format(self.read_stream("ImageInfo/CameraBinning")[0])
            ),
            integration=self.read_stream(
                "ImageInfo/FramesPerImage", XrmDataTypes.XRM_UNSIGNED_INT, strict=False
            )[0],
            read_out_rate=self.read_stream(
                "ImageInfo/ReadoutFreq", XrmDataTypes.XRM_FLOAT, strict=False
            )[0],
            read_out_rate_unit=UnitsFrequency.HERTZ,
            zoom=self.read_stream(
                "ImageInfo/OpticalMagnification", XrmDataTypes.XRM_FLOAT, strict=False
            )[0],
        )

    @txrm_property(fallback=model.Channel(id="Channel:0"))
    def _ome_channel(self):
        return model.Channel(
            id="Channel:0",
            # Energies are 0 for VLM
            acquisition_mode=AcquisitionMode.OTHER
            if self.energies
            else AcquisitionMode.BRIGHT_FIELD,
            illumination_type=IlluminationType.TRANSMITTED,
            light_source_settings=self._ome_light_source_settings,
            detector_settings=self._ome_detector_settings,
            samples_per_pixel=1,
        )

    @txrm_property(fallback=None)
    def _ome_pixels(self):
        # Get image shape
        # number of frames (T in tilt series), Y, X:
        shape = self.output_shape

        # Get metadata variables from ole file:
        exposures = self.exposures.copy()

        pixel_size = self.image_info.get("PixelSize", (0,))[0] * 1.0e3  # micron to nm

        x_positions = [
            coord * 1.0e3 for coord in self.image_info["XPosition"]
        ]  # micron to nm
        y_positions = [
            coord * 1.0e3 for coord in self.image_info["YPosition"]
        ]  # micron to nm
        z_positions = [
            coord * 1.0e3 for coord in self.image_info["ZPosition"]
        ]  # micron to nm

        if self.is_mosaic:
            mosaic_columns, mosaic_rows = self.mosaic_dims
            # Calculates:
            # - Mean exposure, throwing away any invalid 0 values
            # - The centre of the stitched mosaic image (as opposed to the centre of a single tile)
            # Both should be returned as a list to reduce changes to the next section.
            valid_idxs = np.nonzero(exposures)[0]
            exposures = [np.mean(np.asarray(exposures)[valid_idxs])]
            # The mosaic centre is found by taking the first x & y positions (centre of the first tile,
            # which is the bottom-left in the mosaic), taking away the distance between this and the
            # bottom-left corner, then adding the distance to the centre of the mosaic (calculated using pixel size).
            #
            # More verbosely:
            # The physical size of the stitched mosaic is divided by the rows/columns (columns for x, rows for y).
            # This finds the physical size of a single tile. This is then halved, finding in the physical
            # distance (x, y) from between a corner and the centre of a tile. Then this distance is taken from the
            # (x, y) stage coordinates of the first tile to get the stage coordinates of the bottom-left of the mosaic.
            # Half the physical size of the stitched mosaic is added to this, resulting in the in stage coordinates
            # of the mosaic centre.
            x_positions = [
                x_positions[0]
                + (1.0 - 1.0 / mosaic_columns) * (pixel_size * shape[2] / 2.0)
            ]
            y_positions = [
                y_positions[0] + (1.0 - 1.0 / mosaic_rows) * (pixel_size * shape[1] / 2.0)
            ]
            z_positions = [
                np.mean(np.asarray(z_positions)[valid_idxs])
            ]  # Average Z for a stitched mosaic
            # # NOTE: the number of mosaic rows & columns and the pixel size are all written before acquisition but
            # the xy positions are written during, so only the first frame can be relied upon to have an xy
            # position.

        # Run checks to make sure the value lists are long enough
        exp_len_diff = shape[0] - len(exposures)
        if exp_len_diff > 0:
            logging.error(
                "Not enough exposure values for each plane (%i vs %i). Adding zeros to the later planes.",
                len(exposures),
                shape[0],
            )
            for _ in range(exp_len_diff):
                exposures.append(0)

        x_len_diff = shape[0] - len(x_positions)
        if x_len_diff > 0:
            logging.error(
                "Not enough x values for each plane (%i vs %i). Adding zeros to the later planes.",
                len(x_positions),
                shape[0],
            )
            for _ in range(x_len_diff):
                x_positions.append(0)

        y_len_diff = shape[0] - len(y_positions)
        if y_len_diff > 0:
            logging.error(
                "Not enough y values for each plane (%i vs %i). Adding zeros to the later planes.",
                len(y_positions),
                shape[0],
            )
            for _ in range(y_len_diff):
                y_positions.append(0)

        # Add plane/tiffdata for each plane in the stack
        tiffdata_list = []
        plane_list = []
        for count in range(shape[0]):

            tiffdata_list.append(
                model.TiffData(
                    first_c=0, first_t=0, first_z=count, ifd=count, plane_count=1
                )
            )

            plane_list.append(
                model.Plane(
                    the_c=0,
                    the_t=0,
                    the_z=count,
                    delta_t=(self.datetimes[count] - self.datetimes[0]).total_seconds(),
                    delta_t_unit=UnitsTime.SECOND,
                    exposure_time=exposures[count],
                    position_x=x_positions[count],
                    position_x_unit=UnitsLength.NANOMETER,
                    position_y=y_positions[count],
                    position_y_unit=UnitsLength.NANOMETER,
                    position_z=z_positions[count],
                    position_z_unit=UnitsLength.NANOMETER,
                )
            )

        return model.Pixels(
            id="Pixels:0",
            dimension_order="XYCZT",
            size_x=shape[2],
            size_y=shape[1],
            size_c=1,
            size_z=shape[0],
            size_t=1,
            type="uint16",
            physical_size_x=pixel_size,
            physical_size_x_unit=UnitsLength.NANOMETER,
            physical_size_y=pixel_size,
            physical_size_y_unit=UnitsLength.NANOMETER,
            physical_size_z=1,
            physical_size_z_unit=UnitsLength.REFERENCEFRAME,
            tiff_data_blocks=tiffdata_list,
            planes=plane_list,
            channels=[self._ome_channel],
        )

    @txrm_property(fallback=None)
    def _ome_image(self):
        return model.Image(
            id="Image:0",
            acquisition_date=self.datetimes[0],
            description="An OME-TIFF file, converted by txrm2tiff",
            pixels=self._ome_pixels,
            instrument_ref=self._ome_instrument_ref,
            objective_settings=self._ome_objective_ref,
        )

    @txrm_property(fallback=None)
    def metadata(self):
        instruments = []
        print("OME image: ", self._ome_image)
        if self._ome_instrument is not None:
            instruments.append(self._ome_instrument)
        return model.OME(
            creator=f"txrm2tiff {__version__}",
            images=[self._ome_image],
            instruments=instruments,
        )