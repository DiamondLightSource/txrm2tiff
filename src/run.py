from pathlib import Path
from file_sorting import *
import sys


def run(input, custom_reference=None, output=None, ignore_reference=False):
    input_filepath = Path(input)
    if file_can_be_opened(input_filepath) and all_frames_written(input_filepath):

        # If no output is supplied:
        if output is None:
            output_filepath = input_filepath
            if input_filepath.suffix == ".txrm":
                output_filepath.with_suffix(".ome.tiff")
            elif input_filepath.suffix == ".xrm":
                output_filepath.with_suffix(".ome.tif")

        else:
            output_filepath = Path(output)

        TxrmToTiff().convert(input_filepath, output_filepath, custom_reference, ignore_reference)
