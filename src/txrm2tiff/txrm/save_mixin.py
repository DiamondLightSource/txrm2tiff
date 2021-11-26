import logging

import numpy as np
from pathlib import Path
from typing import Optional
from numpy.typing import DTypeLike

from ..utils.metadata import create_ome_metadata, dtype_dict
from ..utils.file_handler import manual_save, manual_annotation_save


class SaveMixin:
    def save_image(
        self,
        filepath: Optional[Path] = None,
        datatype: Optional[DTypeLike] = None,
        flip: bool = True,
        clear_images: bool = True,
        mkdir: bool = False,
    ):
        if filepath is None:
            filepath = self.path.with_suffix(".ome.tiff")
        if not self.referenced:
            logging.info("Saving without reference")

        im = self.get_output(flip, clear_images)
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
            manual_annotation_save(
                filepath.parent / f"{filepath.stem}_Annotated{filepath.suffix}",
                self.annotated_image,
            )

    def create_metadata(self, filepath: Path):
        return create_ome_metadata(self, filepath.stem)
