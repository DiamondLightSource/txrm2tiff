from __future__ import annotations
import traceback
import typing

from txrm2tiff.xradia_properties import stream_dtypes


from .txrm_functions import general
from .xradia_properties import XrmDataTypes
from .utils.exceptions import TxrmFileError

if typing.TYPE_CHECKING:
    from .txrm.abc.images import TxrmWithImages

    TxrmWithImageType = typing.TypeVar("TxrmWithImageType", bound=TxrmWithImages)


class Inspector:

    def __init__(self, txrm: TxrmWithImageType):
        self.txrm = txrm
        self._output_text = ""

    def inspect(self, extra: bool = False) -> None:
        self._output_text += f"\n{self.txrm.name}\t(v{self.txrm.version} file)\n"

        self._output_text += "------------------------------------------------------------------------------------------\n"

        self.inspect_basic()
        self._output_text += "------------------------------------------------------------------------------------------\n\n"

        if extra:
            self.inspect_extra()
            self._output_text += "------------------------------------------------------------------------------------------\n\n"

    def inspect_basic(self) -> None:
        images_taken = typing.cast(
            int, self.txrm.image_info.get("ImagesTaken", (0,))[0]
        )
        num_images = typing.cast(int, self.txrm.image_info.get("NoOfImages", (0,))[0])
        if images_taken != num_images:
            num_str = f"{images_taken} images (of {num_images} planned)"
        else:
            num_str = f"{images_taken} images"
        image_dims = self.txrm.image_dims
        self._output_text += "{0} of type {1} with dimensions: {2}\n".format(
            num_str,
            (
                self.txrm.image_dtype
                if self.txrm.image_dtype is None
                else self.txrm.image_dtype
            ),
            (
                "Unknown"
                if image_dims is None
                else ", ".join([str(i) for i in image_dims])
            ),
        )
        shape = self.txrm.shape
        if shape is not None and shape[::-1] != self.txrm.image_dims:
            # Currently only used for v3 mosaics but may be useful if analysing a processed txrm object
            self._output_text += f"The images are stored as an array of shape (rows x columns): {shape[0]}x{shape[1]}\n"

        mosaic_dims = self.txrm.mosaic_dims
        if self.txrm.is_mosaic:
            if mosaic_dims is None:
                self._output_text += "Is a mosaic of unknown dimensions\n"
            else:
                self._output_text += f"Is a mosaic of shape (rows x coumns): {mosaic_dims[0]}x{mosaic_dims[1]}\n"
        else:
            self._output_text += "Not a mosaic\n"

        self._output_text += "Pixel size: {0}μm\n".format(
            self.txrm.image_info.get("PixelSize", (0,))[0]
        )

        if self.txrm.has_stream("ReferenceData/Image"):
            reference_dims = self.txrm.reference_dims
            self._output_text += (
                "Reference of type {0} applied with dimensions: {1}\n".format(
                    self.txrm.reference_dtype,
                    (
                        "Unknown"
                        if reference_dims is None
                        else ", ".join([str(i) for i in reference_dims])
                    ),
                )
            )
        else:
            self._output_text += "No reference applied\n"

    def inspect_extra(self) -> None:
        self._inspect_image_info()
        self._output_text += "\n"
        self._inspect_reference_info()
        self._output_text += "\n"
        self._inspect_position_info()

    def _inspect_image_info(self) -> None:
        if self.txrm.image_info:
            self._output_text += "ImageInfo streams:\n"
            for name, values in self.txrm.image_info.items():
                self._output_text += "\t{0}: {1}\n\n".format(
                    name, ", ".join([str(p) for p in values])
                )

    def _inspect_reference_info(self) -> None:
        if self.txrm.reference_info:
            self._output_text += "ReferenceData/ImageInfo streams:\n"
            for name, values in self.txrm.reference_info.items():
                self._output_text += "\t{0}: {1}\n\n".format(
                    name, ", ".join([str(p) for p in values])
                )

    def _inspect_position_info(self) -> None:
        if self.txrm.position_info:
            self._output_text += "PositionInfo streams:\n"
            for name, (pos, unit) in self.txrm.position_info.items():
                self._output_text += "\t{0} ({1}): {2}\n\n".format(
                    name, unit, ", ".join([f"{p:.3f}" for p in pos])
                )
            # Displays values to 3dp in Data Explorer

    def _get_stream_list(self) -> list[str]:
        return self.txrm.list_streams()

    def list_streams(self) -> None:
        for stream in self._get_stream_list():
            if self.txrm._ole is None:
                raise TxrmFileError("Unable to access OLE file")
            stream_size = self.txrm._ole.get_size(stream)
            self._output_text += f"{stream:110s} Size: {stream_size:6d} bytes\n"

    def inspect_streams(self, *keys: str) -> None:
        self._output_text += "Inspecting streams: " + ", ".join(keys) + "\n"
        try:
            for key in keys:
                if self.txrm.has_stream(key):

                    self._output_text += f"\n\n{key}:"
                    if key in stream_dtypes.streams_dict:
                        try:
                            dtype = stream_dtypes.streams_dict.get(key)
                            values = general.read_stream(
                                self.txrm._ole,
                                key,
                                dtype,
                                strict=True,
                            )
                            values_str = ", ".join([str(i) for i in values])
                            self._output_text += f"\t{values_str} (stored as {dtype})"
                        except (ValueError, TypeError) as e:
                            self._output_text += (
                                f"\tWARNING: Expected data type unsuccessful:\t{e}\n"
                            )
                            self.inspect_unknown_dtype_stream(key)
                    else:
                        self._output_text += "\nUnknown data type. "
                        self.inspect_unknown_dtype_stream(key)
                else:
                    self._output_text += f"\nStream '{key}' does not exist.\n"
        except Exception as e:
            self._output_text += f"Unexpected exception reading stream {key}:\n\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}\n\n"

    def get_text(self) -> str:
        return self._output_text

    def inspect_unknown_dtype_stream(self, key: str) -> None:
        self._output_text += "Trying all XRM data types:\n"
        for dtype_enum in XrmDataTypes:
            try:
                values_str = ", ".join(
                    [
                        str(i)
                        for i in general.read_stream(
                            self.txrm._ole, key, dtype_enum, strict=True
                        )
                    ]
                )
                self._output_text += f"\t{dtype_enum.name + ':':18s}\t{values_str}\n\n"
            except (ValueError, TypeError) as e:
                self._output_text += f"\t{dtype_enum.name + ':':18s}\t{e}\n\n"
