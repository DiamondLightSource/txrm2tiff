from os import access, R_OK
from pathlib import Path
from olefile import OleFileIO, isOleFile
import logging

from .txrm_wrapper import TxrmWrapper


def file_can_be_opened(path):
    return access(str(path), R_OK)


def ole_file_works(path):
    if (path.suffix == ".txrm") or (path.suffix == ".xrm"):
        if isOleFile(str(path)):
            with OleFileIO(str(path)) as ole_file:
                # Check for reference frame, if required:
                txrm_wrapper = TxrmWrapper()
                number_frames_taken = txrm_wrapper.read_imageinfo_as_int(ole_file, "ImagesTaken")
                expected_number_frames = txrm_wrapper.read_imageinfo_as_int(ole_file, "NoOfImages")
                # Returns true even if all frames aren't written, throwing warning.
                if number_frames_taken != expected_number_frames:
                    logging.warning("%s is an incomplete %s file: only %i out of %i frames have been written",
                                path.name, path.suffix, number_frames_taken, expected_number_frames)
                if not ole_file.exists("ReferenceData/Image"):
                    logging.warning("No reference data found in file %s", path)
                return True
                    
        else:
            logging.warning("Could not read ole file %s", path)
    else:
        logging.warning("%s not .txrm or .xrm", path)
    return False

