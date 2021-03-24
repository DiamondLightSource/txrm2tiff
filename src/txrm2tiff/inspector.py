import logging
from pathlib import Path

from olefile import OleFileIO, isOleFile
import numpy as np
from oxdls import OMEXML

from . import txrm_wrapper

class Inspector(OleFileIO):

    def __init__(self, filename):
        self.fp = None  # Assign fp to avoid errors if invalid OLE file (i.e. not opened)
        path = Path(filename)
        self._filename = path.name
        if path.exists() and isOleFile(filename):
            super().__init__(filename)
            self._output_text = f"\n{self._filename}\n\n"
        else:
            raise IOError("'%s' is not a valid xrm/txrm file" % self._filename)
        if self.fp is None:
            raise IOError("'%s' failed to open for unknown reasons" % self._filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def inspect(self, extra=False):
        self._output_text = f"\n{self._filename}\n\n"

        self._output_text += "{0} images of type '{1}' with dimensions: {2}\n".format(
            txrm_wrapper.extract_number_of_images(self),
            txrm_wrapper.extract_image_dtype(self, "ImageInfo")[0],
            ", ".join([str(i) for i in txrm_wrapper.extract_image_dims(self)])
            )
        self._output_text += "Pixel size: {0}μm\n".format(
            txrm_wrapper.extract_pixel_size(self))
        rows = (
            txrm_wrapper.read_imageinfo_as_int(self, "MosiacRows")
            if self.exists("ImageInfo/MosiacRows") else 0)
        columns = (
            txrm_wrapper.read_imageinfo_as_int(self, "MosiacColumns")
            if self.exists("ImageInfo/MosiacColumns") else 0)
        # Is a mosaic check:
        if rows > 1 or columns > 1:
            self._output_text += f"Is a mosaic of shape (rows x coumns): {rows}x{columns}\n"
        else:
            self._output_text += "Not a mosaic\n"
        if self.exists("ReferenceData/Image"):
            self._output_text += "Reference of type '{0}' applied with dimensions: {1}\n".format(
                txrm_wrapper.extract_image_dtype(self, "ReferenceData")[0],
                ", ".join([str(i) for i in txrm_wrapper.extract_ref_dims(self)])
                )
        else:
            self._output_text += "No reference applied\n"
        self._output_text += "\n"
        
        if extra:
            axis_dict = txrm_wrapper.get_axis_dict(self)

            self._output_text += "Exposures (s): {0}\n\n".format(
                ", ".join([str(i) for i in txrm_wrapper.extract_multiple_exposure_times(self)]))
            self._output_text += "Xray magnification: {0}\n\n".format(
                txrm_wrapper.extract_xray_magnification(self))

            samplex = axis_dict.get(1)
            if samplex:
                self._output_text += "{0} ({1}): {2}\n\n".format(
                    samplex[0], samplex[1],
                    ", ".join([str(i) for i in txrm_wrapper.extract_x_coords(self)]))
            sampley = axis_dict.get(2)
            if sampley:
                self._output_text += "{0} ({1}): {2}\n\n".format(
                    sampley[0], sampley[1],
                    ", ".join([str(i) for i in txrm_wrapper.extract_y_coords(self)]))
            samplez = axis_dict.get(3)
            if samplez:
                self._output_text += "{0} ({1}): {2}\n\n".format(
                    samplez[0], samplez[1],
                    ", ".join([str(i) for i in txrm_wrapper.extract_z_coords(self)]))
            sampletheta = axis_dict.get(4)
            if sampletheta:
                self._output_text += "{0} ({1}): {2}\n\n".format(
                    sampletheta[0], sampletheta[1],
                    ", ".join([str(i) for i in txrm_wrapper.extract_tilt_angles(self)]))
            energy = axis_dict.get(13)
            if energy:
                self._output_text += "{0} ({1}): {2}\n\n".format(
                    energy[0], energy[1],
                    ", ".join([str(i) for i in txrm_wrapper.extract_energies(self)]))
            self._output_text += "Axis units: {0}\n".format(
                    ", ".join([" ".join(pair) for pair in axis_dict.values()]))
            self._output_text += "\n"

    def _get_stream_list(self):
        return ["/".join(stream) for stream in self.listdir(streams=True, storages=False)]
            

    def list_streams(self):
        for stream in self._get_stream_list():
            self._output_text += f"{stream}\n"

    def inspect_streams(self, *keys):
        self._output_text += "Inspecting streams: " + ", ".join(keys) + "\n"
        try:
            for key in keys:
                if self.exists(key):
                    stream = self.openstream(key)
                    stream_binary = stream.read()

                    self._output_text += f"\n\n{key}:\n"
                    self._output_text += f"\tbinary:           \t{stream_binary}\n"
                    self._output_text += f"\thex binary:       \t{stream_binary.hex()}\n"

                    for buff_type in txrm_wrapper.data_type_dict.values():
                        if buff_type[1] is not None:
                            try:
                                values = ",".join(
                                    [str(i) for i in np.frombuffer(stream.getvalue(), buff_type[1]).tolist()])
                                self._output_text += \
                                    f"\t{buff_type[0] + ':':18s}\t{values}\n"
                            except ValueError as e:
                                self._output_text += f"\t{buff_type[0] + ':':18s}\t{e}\n"
                else:
                    self._output_text += f"{key} does not exist\n"
        except Exception as e:
            self._output_text += f"\n{e}\n"

    def get_text(self):
        return self._output_text
