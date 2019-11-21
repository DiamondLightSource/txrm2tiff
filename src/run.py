#!/usr/bin/python
from pathlib import Path
from file_sorting import *
import sys


def main(input, output=None):
    input_filepath = Path(input)
    if output is None:
        output_filepath = input_filepath
        output_filepath.with_suffix(".ome.tiff")
    else:
        output_filepath = Path(output)
    if file_can_be_opened(input_filepath) and all_frames_written(input_filepath):
        TxrmToTiff.convert(input_filepath, output_filepath)


if __name__ == "__main__":
    main(sys.argv[1:])
