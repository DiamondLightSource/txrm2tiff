import traceback
from dataclasses import dataclass

from txrm2tiff.xradia_properties import stream_dtypes

from .txrm.abstract import AbstractTxrm
from .txrm_functions import general
from .xradia_properties import XrmDataTypes


class Inspector:

    def __init__(self, txrm:AbstractTxrm):
        self.txrm = txrm
        self._output_text = ""

    def inspect(self, extra=False):
        self._output_text += f"\n{self.txrm.name}\t(v{self.txrm.version} file)\n"

        self._output_text += "------------------------------------------------------------------------------------------\n"

        self.inspect_basic()
        self._output_text += "------------------------------------------------------------------------------------------\n\n"

        if extra:
            self.inspect_extra()
            self._output_text += "------------------------------------------------------------------------------------------\n\n"

    def inspect_basic(self):
        images_taken = self.txrm.image_info.get("ImagesTaken", (0,))[0]
        num_images = self.txrm.image_info.get("NoOfImages", (0,))[0]
        if images_taken != num_images:
            num_str = f"{images_taken} images (of {num_images} planned)"
        else:
            num_str = f"{images_taken} images"
        self._output_text += "{0} of type {1} with dimensions: {2}\n".format(
            num_str,
            self.txrm.image_dtype
            if self.txrm.image_dtype is None
            else self.txrm.image_dtype,
            ", ".join([str(i) for i in self.txrm.image_dims]),
        )
        if self.txrm.shape[::-1] != self.txrm.image_dims:
            # Currently only used for v3 mosaics but may be useful if analysing a processed txrm object
            self._output_text += f"The images are stored as an array of shape (rows x columns): {self.txrm.shape[0]}x{self.txrm.shape[1]}\n"

        if self.txrm.is_mosaic:
            self._output_text += f"Is a mosaic of shape (rows x coumns): {self.txrm.mosaic_dims[0]}x{self.txrm.mosaic_dims[1]}\n"
        else:
            self._output_text += "Not a mosaic\n"

        self._output_text += "Pixel size: {0}μm\n".format(
            self.txrm.image_info.get("PixelSize", (0,))[0]
        )

        if self.txrm.has_stream("ReferenceData/Image"):
            self._output_text += (
                "Reference of type {0} applied with dimensions: {1}\n".format(
                    self.txrm.reference_dtype,
                    ", ".join([str(i) for i in self.txrm.reference_dims]),
                )
            )
        else:
            self._output_text += "No reference applied\n"

    def inspect_extra(self):
        self._inspect_image_info()
        self._output_text += "\n"
        self._inspect_reference_info()
        self._output_text += "\n"
        self._inspect_position_info()

    def _inspect_image_info(self):
        if self.txrm.image_info:
            self._output_text += "ImageInfo streams:\n"
            for name, values in self.txrm.image_info.items():
                self._output_text += "\t{0}: {1}\n\n".format(
                    name, ", ".join([f"{p}" for p in values])
                )

    def _inspect_reference_info(self):
        if self.txrm.reference_info:
            self._output_text += "ReferenceData/ImageInfo streams:\n"
            for name, values in self.txrm.reference_info.items():
                self._output_text += "\t{0}: {1}\n\n".format(
                    name, ", ".join([f"{p}" for p in values])
                )

    def _inspect_position_info(self):
        if self.txrm.position_info:
            self._output_text += "PositionInfo streams:\n"
            for name, (pos, unit) in self.txrm.position_info.items():
                self._output_text += "\t{0} ({1}): {2}\n\n".format(
                    name, unit, ", ".join([f"{p:.3f}" for p in pos])
                )
            # Displays values to 3dp in Data Explorer

    def _get_stream_list(self):
        return self.txrm.list_streams()

    def list_streams(self):
        for stream in self._get_stream_list():
            stream_size = self.txrm.ole.get_size(stream)
            self._output_text += f"{stream:110s} Size: {stream_size:6d} bytes\n"

    def inspect_streams(self, *keys):
        self._output_text += "Inspecting streams: " + ", ".join(keys) + "\n"
        try:
            for key in keys:
                if self.txrm.has_stream(key):

                    self._output_text += f"\n\n{key}:"
                    if key in stream_dtypes.streams_dict:
                        try:
                            dtype = stream_dtypes.streams_dict.get(key)
                            values = general.read_stream(
                                self.txrm.ole,
                                key,
                                dtype,
                                strict=True,
                            )
                            values_str = ", ".join([str(i) for i in values])
                            self._output_text += f"\t{values_str} (stored as {dtype})"
                        except (ValueError, TypeError) as e:
                            self._output_text += (
                                f"\tWARNING: Expected data type unsucessful:\t{e}\n"
                            )
                            self.inspect_unknown_dtype_stream(key)
                    else:
                        self._output_text += f"\nUnknown data type. "
                        self.inspect_unknown_dtype_stream(key)
                else:
                    self._output_text += f"\nStream '{key}' does not exist.\n"
        except Exception as e:
            self._output_text += f"Unexpected exception reading stream {key}:\n\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}\n\n"

    def get_text(self):
        return self._output_text

    def inspect_unknown_dtype_stream(self, key: str) -> None:
        self._output_text += "Trying all XRM data types:\n"
        for dtype_enum in XrmDataTypes:
            try:
                values_str = ", ".join(
                    [
                        str(i)
                        for i in general.read_stream(
                            self.txrm.ole, key, dtype_enum, strict=True
                        )
                    ]
                )
                self._output_text += f"\t{dtype_enum.name + ':':18s}\t{values_str}\n\n"
            except (ValueError, TypeError) as e:
                self._output_text += f"\t{dtype_enum.name + ':':18s}\t{e}\n\n"
