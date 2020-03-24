from pathlib import Path
from file_sorting import *
from logger import create_logger


def run(input, custom_reference=None, output=None, ignore_reference=False, logging_level="info"):
    create_logger(logging_level.lower())

    input_filepath = Path(input)
    logging.info(f"Running txrm2tiff on {input_filepath.name}")
    if file_can_be_opened(input_filepath) and ole_file_works(input_filepath):

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
