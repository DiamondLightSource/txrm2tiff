import logging

from pathlib import Path
from typing import Optional
import numpy as np
from numpy.typing import DTypeLike

from ..utils.metadata import create_ome_metadata, dtype_dict
from ..utils.file_handler import manual_save


class SaveMixin:
    def save_images(
        self,
        filepath: Optional[Path] = None,
        datatype: Optional[DTypeLike] = None,
        shifts: bool = False,
        flip: bool = False,
        clear_images: bool = False,
        mkdir: bool = False,
        save_annotations: bool = True,
        annotated_path: Optional[Path] = None,
        strict: Optional[bool] = None,
    ) -> bool:
        """Saves images (if available) returning True if successful."""
        if strict is None:
            strict = self.strict
        try:
            if filepath is None:
                if self.path is None:
                    raise ValueError(
                        "An output filepath must be given if an input path was not given."
                    )
                filepath = self.path.resolve().with_suffix(".ome.tiff")
            if not self.referenced:
                logging.info("Saving without reference")

            im = self.get_output(
                load=True, shifts=shifts, flip=flip, clear_images=clear_images
            )
            if im is None:
                raise AttributeError("Cannot save image as no image has been loaded.")
            if datatype is not None:
                datatype = np.dtype(datatype)
                if datatype.name not in dtype_dict:
                    datatype = None
                    logging.warning(
                        "Invalid data type '%s', must be %s or None. Defaulting to saving as %s",
                        datatype,
                        ", ".join(dtype_dict.keys()),
                        str(im.dtype),
                    )
            metadata = self.create_metadata(filepath)

            if mkdir:
                filepath.parent.mkdir(parents=True, exist_ok=True)

            manual_save(filepath, im, datatype, metadata)
            if (
                save_annotations
                and hasattr(self, "annotate")
                and self.annotated_image is not None
            ):
                if annotated_path is None:
                    # Generate default path
                    filename = filepath.name
                    if filename.lower().endswith(".ome.tiff"):
                        # Special case for ome.tiff
                        stem, suffix = filename.rsplit(".ome.", 1)
                    else:
                        stem, suffix = filename.rsplit(".", 1)
                    annotated_path = filepath.parent / f"{stem}_Annotated.{suffix}"
                self.save_annotations(annotated_path, mkdir=mkdir, strict=strict)
            return True
        except Exception:
            logging.error("Saving failed", exc_info=not strict)
            if strict:
                raise
            return False

    def create_metadata(self, filepath: Path):
        return create_ome_metadata(self, filepath.stem)
