#!/usr/bin/python

import numpy as np
from itertools import takewhile
from scipy.constants import h, c, e
import logging


class TxrmWrapper:

    # Returns a list of type "dtype".
    def read_stream(self, ole, key, dtype):
        if ole.exists(key):
            stream = ole.openstream(key)
            # tolist() is needed or the data will be kept in numpy dtypes, which ISPyB can't handle
            return np.fromstring(stream.getvalue(), dtype).tolist()
        else:
            logging.error("Stream %s does not exist in ole file", key)

    def extract_unknown_dtype_image_data(img_stream_bytes, img_size, num_mosaic_tiles=1):
        img_stream_length = len(img_stream_bytes)
        if (img_stream_length == ((img_size * 2) / num_mosaic_tiles)):
            return np.fromstring(img_stream_bytes, dtype=np.uint16)
        elif (img_stream_length == ((img_size * 4) / num_mosaic_tiles)):
            return np.fromstring(img_stream_bytes, dtype=np.float32)
        else:
            logging.error("Unexpected data type with %f bytes per pixel",
                          (ref_stream_length * num_mosaic_tiles / img_size))
            return None

    def extract_single_image(self, ole, numimage, numrows, numcols):
        # Read the images - They are stored in the txrm as ImageData1 ...
        # Each folder contains 100 images 1-100, 101-200
        img_key = "ImageData%i/Image%i" % \
            (np.ceil(numimage / 100.0), numimage)
        if ole.exists(img_key):
            img_stream_bytes = ole.openstream(img_key).getvalue()
            img_size = numrows * numcols
            imgdata = self.extract_unknown_dtype_image_data(img_stream_bytes, img_size)
            if imagedata is None:
                raise TypeError("Image is stored as unexpected type. Expecting uint16 or float32.")
            imgdata.shape = (numrows, numcols)
            return imgdata
        else:
            return np.zeros([])

    def read_imageinfo_as_int(self, ole, key_part, ref_data=False):
        stream_key = f"ImageInfo/{key_part}"
        if ref_data:
            stream_key = "ReferenceData/" + stream_key
        integer_tuple = self.read_stream(ole, stream_key, "i")
        if integer_tuple is not None:
            return integer_tuple[0]

    def extract_image_dims(self, ole):
        return (self.read_imageinfo_as_int(ole, "ImageHeight"),
                self.read_imageinfo_as_int(ole, "ImageWidth"))

    def extract_ref_dims(self, ole):
        if ole.exists("ReferenceData/ImageInfo/ImageHeight"):
            return (self.read_imageinfo_as_int(ole, "ImageHeight", ref_data=True),
                    self.read_imageinfo_as_int(ole, "ImageWidth", ref_data=True))
        return self.extract_image_dims(ole)

    def extract_number_of_images(self, ole):
        return self.read_imageinfo_as_int(ole, "NoOfImages")

    def extract_tilt_angles(self, ole):
        return self.read_stream(ole, "ImageInfo/Angles", "f")

    def extract_x_coords(self, ole):
        return np.asarray(self.read_stream(ole, "ImageInfo/XPosition", "f"))

    def extract_y_coords(self, ole):
        return np.asarray(self.read_stream(ole, "ImageInfo/YPosition", "f"))

    def extract_exposure_time(self, ole):
        if ole.exists('ImageInfo/ExpTimes'):
            # Returns the exposure of the image at the closest angle to 0 degrees:
            exposures = self.read_stream(ole, "ImageInfo/ExpTimes", "f")
            absolute_angles = np.array([abs(angle) for angle in self.extract_tilt_angles(ole)])
            min_angle = absolute_angles.argmin()
            return exposures[min_angle]
        elif ole.exists("ImageInfo/ExpTime"):
            return self.read_stream(ole, "ImageInfo/ExpTime", dtype="f")[0]
        else:
            raise Exception("No exposure time available in ole file.")

    def extract_multiple_exposure_times(self, ole):
        if ole.exists('ImageInfo/ExpTimes'):
            # Returns the exposure of the image at the closest angle to 0 degrees:
            exposures = self.read_stream(ole, "ImageInfo/ExpTimes", "f")
            return exposures
        elif ole.exists("ImageInfo/ExpTime"):
            return self.read_stream(ole, "ImageInfo/ExpTime", dtype="f")
        else:
            raise Exception("No exposure time available in ole file.")

    def extract_pixel_size(self, ole):
        return self.read_stream(ole, 'ImageInfo/PixelSize', "f")[0]

    def extract_all_images(self, ole):
        (num_rows, num_columns) = self.extract_image_dims(ole)
        if ole.exists("ImageInfo/ExpTimes"):
            images_taken = self.read_imageinfo_as_int(ole, "ImagesTaken")
        elif ole.exists("ImageInfo/ExpTime"):
            images_taken = 1
        if (num_rows * num_columns > 0):
            # Iterates through images until the number of images taken
            # lambda check has been left in in case stream is wrong
            images = (self.extract_single_image(ole, i, num_rows, num_columns) for i in range(1, images_taken + 1))
            return list(takewhile(lambda image: image.size > 1, images))
        else:
            raise AttributeError("No image dimensions found")

    def extract_first_image(self, ole):
        return self.extract_all_images(ole)[0]

    def extract_xray_magnification(self, ole):
        return self.read_stream(ole, "ImageInfo/XrayMagnification", "f")[0]

    def extract_energy(self, ole):
        energies = self.read_stream(ole, "ImageInfo/Energy", "f")
        return np.mean(energies)

    def extract_wavelength(self, ole):
        return (h * c) / (self.extract_energy(ole) * e)

    def create_reference_mosaic(self, ole, refdata, image_rows, image_columns, mosaic_rows, mosaic_columns):
        ref_num_rows = image_rows // mosaic_rows
        ref_num_columns = image_columns // mosaic_columns
        refdata.shape = (ref_num_rows, ref_num_columns)
        return np.tile(refdata, (mosaic_rows, mosaic_columns))

    def rescale_ref_exposure(self, ole, refdata):
        # Normalises the reference exposure
        # (if it is a tomo, it does this relative to the 0 degree image)
        ref_exp = self.read_stream(ole, "ReferenceData/ExpTime", dtype="f")[0]
        im_exp = self.extract_exposure_time(ole)
        ref_exp_rescale = im_exp / ref_exp
        if ref_exp_rescale == 1:
            return refdata
        return refdata * ref_exp_rescale

    def extract_reference_image(self, ole):
        # Read the reference image.
        # Reference info is stored under 'ReferenceData/...'
        (num_rows, num_columns) = self.extract_ref_dims(ole)
        ref_stream_bytes = ole.openstream("ReferenceData/Image").getvalue()
        img_size = num_rows * num_columns

        #  In XMController 13+ the reference file is not longer always a float. Additionally, there
        #  are multiple methods to apply a reference now, so this has been kept general, rather than
        #  being dependent on the file version or software version (software version is new metadata
        #  in to XMController 13).
        if ole.exists("ImageInfo/MosiacMode") and self.read_imageinfo_as_int(ole, "MosiacMode") == 1:
        # This MosiacMode has been removed in v13 but is kept in for backward compatibility:
        # ImageInfo/MosiacMode (genuinely how it's spelled in the file) == 1 if it is a mosaic, 0 if not.
            mosaic_rows = self.read_imageinfo_as_int(ole, "MosiacRows")
            mosaic_cols = self.read_imageinfo_as_int(ole, "MosiacColumns")
            mosaic_size_multiplier = mosaic_rows * mosaic_cols
            refdata = self.extract_unknown_dtype_image_data(ref_stream_bytes, img_size, mosaic_size_multiplier)
            if refdata is None:
                raise TypeError("Reference is stored as unexpected type. Expecting uint16 or float32.")
            refdata = self.create_reference_mosaic(ole, refdata, num_rows, num_columns, mosaic_rows, mosaic_cols)
        else:
            refdata = self.extract_unknown_dtype_image_data(img_stream_bytes, img_size)
            if refdata is None:
                raise TypeError("Reference is stored as unexpected type. Expecting uint16 or float32.")
        # num_rows and num_columns are from the array shape of the main image (the stitched image if a mosaic)
        refdata.shape = (num_rows, num_columns)
        return self.rescale_ref_exposure(ole, refdata)
