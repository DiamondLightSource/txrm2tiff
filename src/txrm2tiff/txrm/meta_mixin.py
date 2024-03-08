from ome_types import model

from .txrm_property import txrm_property
from ..xradia_properties.enums import (
    XrmDataTypes,
    XrmObjectiveType,
    XrmSourceType,
)


class MetaMixin:
    @txrm_property(fallback=0)
    def configured_camera_count(self):
        return self.read_stream(
            "ConfigureBackup/ConfigCamera/NumberOfCamera", XrmDataTypes.XRM_UNSIGNED_INT
        )[0]

    @txrm_property(fallback=[])
    def configured_detectors(self):
        return [self._get_detector(i) for i in range(self.configured_camera_count)]

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
    def detector(self):
        camera_idx = (
            self.read_stream("ImageInfo/CameraNo", XrmDataTypes.XRM_UNSIGNED_INT)[0] - 1
        )  # Stream counts from 1
        return self.configured_detectors[camera_idx]

    @txrm_property(fallback=None)
    def microscope(self):
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
    def configured_objectives(self):
        return [
            obj
            for i in range(self.configured_camera_count)
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
    def instrument(self):
        return model.Instrument(
            id="Instrument:0",
            detectors=self.configured_detectors,
            microscope=self.microscope,
            objectives=self.configured_objectives,
            light_source_group=self.configured_light_sources,
        )

    @txrm_property(fallback=None)
    def objective(self):
        camera_idx = (
            self.read_stream("ImageInfo/CameraNo", XrmDataTypes.XRM_UNSIGNED_INT)[0] - 1
        )  # Stream counts from 1
        return self.configured_objectives[camera_idx]

    @txrm_property(fallback=0)
    def configured_source_count(self):
        return self.read_stream(
            "ConfigureBackup/ConfigSources/NumberOfSources",
            XrmDataTypes.XRM_UNSIGNED_INT,
        )[0]

    @txrm_property(fallback=[])
    def configured_light_sources(self):
        return [self._get_light_source(i) for i in range(self.configured_source_count)]

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
    def light_source(self):
        source_idx = self.read_stream(
            "AcquisitionSettings/SourceIndex", XrmDataTypes.XRM_UNSIGNED_INT
        )[
            0
        ]  # Stream counts from 0
        return self.configured_light_sources[source_idx]
