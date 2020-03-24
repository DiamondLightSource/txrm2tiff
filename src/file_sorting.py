from pathlib import Path
from olefile import OleFileIO, isOleFile
import logging

from txrm_wrapper import TxrmWrapper
from txrm2image import TxrmToTiff


def file_can_be_opened(path):
    try:
        open(str(path)).close()
    except IOError as e:
        print(e)
        return False
    return True


def ole_file_works(path):
    if (path.suffix == ".txrm") or (path.suffix == ".xrm"):
        if isOleFile(str(path)):
            ole_file = OleFileIO(str(path))
            # Check for reference frame, if required:
            if (ole_file.exists("ReferenceData/Image")):
                txrm_wrapper = TxrmWrapper()
                number_frames_taken = txrm_wrapper.read_imageinfo_as_int(ole_file, "ImagesTaken")
                expected_number_frames = txrm_wrapper.read_imageinfo_as_int(ole_file, "NoOfImages")
                # Returns true even if all frames aren't written, throwing warning.
                if number_frames_taken != expected_number_frames:
                    logging.warn("This is an incomplete %s file: only %i out of %i frames have been written",
                                 path.suffix, number_frames_taken, expected_number_frames)
                return True
        else:
            logging.warn("Could not read ole file %s", path)
    else:
        logging.warn("%s not .txrm or .xrm", path)
    return False

