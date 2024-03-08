from __future__ import annotations
import logging
from scipy import constants
import numpy as np
from xml.etree import ElementTree
from ome_types import model
from ome_types.model.simple_types import (
    UnitsLength,
    UnitsFrequency,
    UnitsTime,
    Binning,
)
from typing import TYPE_CHECKING

from ..info import __version__
from .txrm_property import txrm_property
from ..xradia_properties.enums import (
    XrmDataTypes,
    XrmSourceType,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class MetaMixin:
    @txrm_property(fallback=0)
    def _ome_configured_camera_count(self) -> int:
        return self.read_stream(
            "ConfigureBackup/ConfigCamera/NumberOfCamera", XrmDataTypes.XRM_UNSIGNED_INT
        )[0]

    @txrm_property(fallback=dict())
    def _ome_configured_detectors(self) -> dict[int, model.Detector]:
        return {
            self._camera_ids[i]: self._get_detector(i)
            for i in range(self._ome_configured_camera_count)
        }

    def _get_detector(self, index: int) -> model.Detector:
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

    @txrm_property(fallback=[])
    def _camera_ids(self) -> list[int]:
        return [
            self._get_camera_id(i) for i in range(self._ome_configured_camera_count)
        ]

    def _get_camera_id(self, index: int) -> int:
        stream_stem = f"ConfigureBackup/ConfigCamera/Camera {index + 1}"
        return self.read_stream(
            f"{stream_stem}/CameraID", XrmDataTypes.XRM_UNSIGNED_INT
        )[0]

    @txrm_property(fallback=None)
    def _ome_detector(self) -> model.Detector:
        camera_id = self.read_stream(
            "ImageInfo/CameraNo", XrmDataTypes.XRM_UNSIGNED_INT
        )[
            0
        ]  # Stream counts from 1
        return self._ome_configured_detectors[camera_id]

    @txrm_property(fallback=None)
    def _ome_microscope(self) -> model.Microscope:
        kwargs = {}
        id_stream = "ConfigureBackup/XRMConfiguration/MachineID"
        if self.has_stream(id_stream):
            machine_id = self.read_stream(id_stream, dtype=XrmDataTypes.XRM_STRING)
            if machine_id:
                kwargs["model"] = machine_id[0]

        return model.Microscope(
            type="Other",
            manufacturer="Xradia",
            **kwargs,
        )

    @txrm_property(fallback=dict())
    def _ome_configured_objectives(self) -> dict[int, dict[str, model.Objective]]:
        return {
            self._get_camera_id(i): self._get_objectives(i)
            for i in range(self._ome_configured_camera_count)
        }

    def _get_objectives(self, index: int) -> dict[str, model.Objective]:
        stream_stem = f"ConfigureBackup/ConfigCamera/Camera {index + 1}"
        id_stream = f"{stream_stem}/ConfigObjectives/ObjectiveID"
        objective_ids = self.read_stream(id_stream, XrmDataTypes.XRM_STRING)
        magnifications = self.read_stream(
            f"{stream_stem}/ConfigObjectives/OpticalMagnification",
            XrmDataTypes.XRM_FLOAT,
        )
        objective_names = self.read_stream(
            f"{stream_stem}/ConfigObjectives/ObjectiveName", XrmDataTypes.XRM_STRING
        )
        zoneplate_names = self.read_stream(
            f"{stream_stem}/ConfigZonePlates/Name", XrmDataTypes.XRM_STRING
        )
        if not zoneplate_names:  # Fallback if zoneplate names fail
            return {
                "objective_name": model.Objective(
                    id=f"Objective:{obj_id}.0",
                    nominal_magnification=magnification,
                    model=objective_name,
                )
                for obj_id, objective_name, magnification in zip(
                    objective_ids, objective_names, magnifications
                )
            }

        return {
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

    @txrm_property(fallback=None)
    def _ome_instrument(self) -> model.Instrument:
        objectives = [
            obj
            for cameras in self._ome_configured_objectives.values()
            for objectives in cameras.values()
            for item in (
                objectives.values() if isinstance(objectives, dict) else [objectives]
            )
            for obj in item  # Cope with either dict or Objective instance
        ]
        return model.Instrument(
            id="Instrument:0",
            detectors=list(self._ome_configured_detectors.values()),
            microscope=self._ome_microscope,
            objectives=objectives,
            light_source_group=self._ome_configured_light_sources,
        )

    @txrm_property(fallback=None)
    def _ome_instrument_ref(self) -> model.InstrumentRef:
        if self._ome_instrument is None:
            logging.info("No instrument to reference")
            return None
        return model.InstrumentRef(id=self._ome_instrument.id)

    @txrm_property(fallback=None)
    def _ome_objective(self) -> model.Objective:
        camera_id = self.read_stream(
            "ImageInfo/CameraNo", XrmDataTypes.XRM_UNSIGNED_INT
        )[0]
        obj_name = self.read_stream("ImageInfo/ObjectiveName", XrmDataTypes.XRM_STRING)[
            0
        ]
        zp_name = self.read_stream("ImageInfo/ZonePlateName", XrmDataTypes.XRM_STRING)
        if zp_name:
            obj_name = f"{obj_name}_{zp_name}"
        objective = self._get_objectives[camera_id][obj_name]
        return objective

    @txrm_property(fallback=None)
    def _ome_objective_settings(self) -> model.ObjectiveSettings:
        if self._ome_objective is None:
            logging.info("No objective to reference")
            return None
        return model.ObjectiveSettings(id=self._ome_objective.id)

    @txrm_property(fallback=0)
    def _ome_configured_source_count(self) -> int:
        return self.read_stream(
            "ConfigureBackup/ConfigSources/NumberOfSources",
            XrmDataTypes.XRM_UNSIGNED_INT,
        )[0]

    @txrm_property(fallback=[])
    def _ome_configured_light_sources(self) -> list[model.LightSource]:
        return [
            self._get_light_source(i) for i in range(self._ome_configured_source_count)
        ]

    def _get_light_source(self, index: int) -> model.LightSource:
        stream_stem = (
            f"ConfigureBackup/ConfigSources/Source {index + 1}"  # Stream from 1
        )
        source_type = XrmSourceType(
            self.read_stream(f"{stream_stem}/Type", XrmDataTypes.XRM_UNSIGNED_INT)[0]
        )
        id_ = f"LightSource:{index}"
        name = self.read_stream(f"{stream_stem}/SourceName", XrmDataTypes.XRM_STRING)[0]
        kwargs = {}
        if source_type == XrmSourceType.XRM_VISUAL_LIGHT_SOURCE:
            source = model.LightEmittingDiode
        else:
            source = model.GenericExcitationSource

            m = []
            current, current_units = self.position_info.get("Current", [[], None])
            if current and current_units:
                m.append(model.Map.M(k="Current", value=", ".join(map(str, current))))
                m.append(model.Map.M(k="CurrentUnits", value=str(current_units)))
            if m:
                kwargs["map"] = model.Map(k="BeamProperties", m=m)

        return source(id=id_, model=name, **kwargs)

    @txrm_property(fallback=None)
    def _ome_light_source(self) -> model.LightSource:
        source_idx = self.read_stream(
            "AcquisitionSettings/SourceIndex", XrmDataTypes.XRM_UNSIGNED_INT
        )[
            0
        ]  # Stream counts from 0
        return self._ome_configured_light_sources[source_idx]

    @txrm_property(fallback=None)
    def _ome_light_source_settings(self) -> model.LightSourceSettings | None:
        if self._ome_light_source is None:
            logging.info("No light source to reference")
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
    def _ome_detector_settings(self) -> model.DetectorSettings:
        kwargs = {}
        if self.has_stream("ImageInfo/CameraBinning"):
            binning_str = "{0}x{0}".format(
                self.read_stream("ImageInfo/CameraBinning")[0]
            )
            kwargs["binning"] = (
                Binning(binning_str) if binning_str in Binning else Binning.OTHER
            )
        return model.DetectorSettings(
            id=self._ome_detector.id,
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
            **kwargs,
        )

    @txrm_property(fallback=model.Channel(id="Channel:0"))
    def _ome_channel(self) -> model.Channel:
        return model.Channel(
            id="Channel:0",
            # Energies are 0 for VLM
            acquisition_mode=(
                model.Channel_AcquisitionMode.OTHER
                if self.energies
                else model.Channel_AcquisitionMode.BRIGHT_FIELD
            ),
            illumination_type=model.Channel_IlluminationType.TRANSMITTED,
            # light_source_settings=self._ome_light_source_settings,
            detector_settings=self._ome_detector_settings,
            samples_per_pixel=1,
        )

    @txrm_property(fallback=None)
    def _ome_pixels(self) -> model.Pixels:
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
                y_positions[0]
                + (1.0 - 1.0 / mosaic_rows) * (pixel_size * shape[1] / 2.0)
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
    def _ome_image(self) -> model.Image:
        kwargs = {}
        if self._ome_modulo is not None:
            kwargs["annotation_ref"] = [model.AnnotationRef(id=self._ome_modulo.id)]
        return model.Image(
            id="Image:0",
            acquisition_date=self.datetimes[0],
            description=f"An OME-TIFF file converted from {self.name}",
            pixels=self._ome_pixels,
            instrument_ref=self._ome_instrument_ref,
            objective_settings=self._ome_objective_settings,
            **kwargs,
        )

    @txrm_property(fallback=None)
    def metadata(self) -> model.OME:
        kwargs = {}
        if self._ome_instrument is not None:
            # If ome instrument metadata fails due to a bad config,
            # it's still worth appending the image info
            kwargs["instruments"] = [self._ome_instrument]
        if self._ome_modulo is not None:
            kwargs["structured_annotations"] = [self._ome_modulo]
        return model.OME(
            creator=f"txrm2tiff {__version__}", images=[self._ome_image], **kwargs
        )

    @txrm_property(fallback=None)
    def _ome_modulo(self) -> model.XMLAnnotation:
        el = ElementTree.Element(
            "Modulo",
            namespace="http://www.openmicroscopy.org/Schemas/Additions/2011-09",
        )
        angles = self.image_info.get("Angles", [])
        if angles and np.sum(angles):
            self._add_angle_subelement(el, angles)
        if self.energies and np.sum(self.energies):
            self._add_energy_subelement(el, self.energies)
        if not el.getchildren:
            # If no modulos were added, don't add the annotation
            return None
        return model.XMLAnnotation(
            id="Annotation:0",
            value=el,
            namespace="openmicroscopy.org/omero/dimension/modulo",
        )

    def _add_energy_subelement(
        self, element: ElementTree.Element, energies: Iterable[float]
    ) -> None:
        sub_el = ElementTree.SubElement(
            element, "ModuloAlongZ", Type="other", TypeDescription="energy", Unit="eV"
        )
        for eng in energies:
            label_sub_el = ElementTree.SubElement(sub_el, "Label")
            label_sub_el.text = str(eng)

    def _add_angle_subelement(
        self, element: ElementTree.Element, angles: Iterable[float]
    ) -> None:
        sub_el = ElementTree.SubElement(
            element, "ModuloAlongZ", Type="angle", Unit="degree"
        )
        for ang in angles:
            label_sub_el = ElementTree.SubElement(sub_el, "Label")
            label_sub_el.text = str(ang)
