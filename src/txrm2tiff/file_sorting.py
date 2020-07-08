from os import access, R_OK
from pathlib import Path
from olefile import OleFileIO, isOleFile
import logging

from .txrm_wrapper import read_imageinfo_as_int



def file_can_be_opened(path):
    if access(str(path), R_OK):
        return True
    logging.error("File %s cannot be opened", path)
    return False


def ole_file_works(path):
    if (path.suffix == ".txrm") or (path.suffix == ".xrm"):
        if isOleFile(str(path)):
            with OleFileIO(str(path)) as ole_file:
                number_frames_taken = read_imageinfo_as_int(ole_file, "ImagesTaken")
                expected_number_frames = read_imageinfo_as_int(ole_file, "NoOfImages")
                # Returns true even if all frames aren't written, throwing warning.
                if number_frames_taken != expected_number_frames:
                    logging.warning("%s is an incomplete %s file: only %i out of %i frames have been written",
                                path.name, path.suffix, number_frames_taken, expected_number_frames)
                # Check for reference frame:
                if not ole_file.exists("ReferenceData/Image"):
                    logging.warning("No reference data found in file %s", path)
                return True
        else:
            logging.warning("Could not read ole file %s", path)
    else:
        logging.warning("%s not .txrm or .xrm", path)
    return False

