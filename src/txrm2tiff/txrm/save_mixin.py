import logging

import numpy as np
from pathlib import Path
from typing import Optional
from numpy.typing import DTypeLike

from ..utils.metadata import create_ome_metadata, dtype_dict
from ..utils.file_handler import manual_save, manual_annotation_save


class SaveMixin:
    def save_images(
        self,
        filepath: Optional[Path] = None,
        datatype: Optional[DTypeLike] = None,
        shifts: bool = True,
        flip: bool = False,
        clear_images: bool = False,
        mkdir: bool = False,
    ) -> bool:
        """Saves images (if available) returning True if successful."""
        try:
            if filepath is None:
                if self.path is None:
                    raise ValueError(
                        "An output filepath must be given if an input path was not given."
                    )
                filepath = self.path.with_suffix(".ome.tiff")
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
                tiff_dir = filepath.resolve().parent
                tiff_dir.mkdir(parents=True, exist_ok=True)

            manual_save(filepath, im, datatype, metadata)
            if self.annotated_image is not None:
                stem = filepath.stem
                suffix = filepath.suffix
                if stem.lower().endswith(".ome") and suffix.lower() == ".tiff":
                    # Special case for ome.tiff
                    suffix = f"{stem[-4:]}{suffix}"
                    stem = stem[:-4]
                manual_annotation_save(
                    filepath.parent / f"{stem}_Annotated{suffix}",
                    self.annotated_image,
                )
            return True
        except Exception:
            logging.error("Saving failed", exc_info=True)
            return False

    def create_metadata(self, filepath: Path):
        return create_ome_metadata(self, filepath.stem)
