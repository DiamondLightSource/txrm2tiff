#!/usr/bin/python

import numpy as np
from itertools import takewhile
from scipy.constants import h, c, e
import logging

data_type_dict = {
    1: ("XRM_BIT", None),
    2:  ("XRM_CHAR", np.byte),
    3:  ("XRM_UNSIGNED_CHAR", np.ubyte),
    4:  ("XRM_SHORT", np.short),
    5:  ("XRM_UNSIGNED_SHORT", np.ushort),
    6:  ("XRM_INT", np.intc),
    7:  ("XRM_UNSIGNED_INT", np.uintc),
    8:  ("XRM_LONG", np.int_),
    9:  ("XRM_UNSIGNED_LONG", np.uint),
    10: ("XRM_FLOAT", np.single),
    11: ("XRM_DOUBLE", np.double),
    12: ("XRM_STRING", np.str_),
    13: ("XRM_DATATYPE_SIZE", None)
}


def extract_image_dtype(ole, key_part):
    key = f"{key_part}/DataType"
    integer_list = read_stream(ole, key, "i")
    if integer_list is not None:
        return data_type_dict.get(integer_list[0], (None, None))
    logging.error("Stream %s does not exist in ole file", key)
    return (None, None)


# Returns a list of type "dtype".
def read_stream(ole, key, dtype):
    if ole.exists(key):
        stream_value = ole.openstream(key).getvalue()
        return np.frombuffer(stream_value, dtype).tolist()
    logging.error("Stream %s does not exist in ole file", key)
    return None


def read_imageinfo_as_int(ole, key_part, ref_data=False):
    stream_key = f"ImageInfo/{key_part}"
    if ref_data:
        stream_key = "ReferenceData/" + stream_key
    integer_list = read_stream(ole, stream_key, "i")
    if integer_list is not None:
        return integer_list[0]


def extract_single_image(ole, numimage, numrows, numcols):
    # Read the images - They are stored in the txrm as ImageData1 ...
    # Each folder contains 100 images 1-100, 101-200
    img_key = f"ImageData{int(np.ceil(numimage / 100.0))}/Image{numimage}"
    if ole.exists(img_key):
        img_stream_bytes = ole.openstream(img_key).getvalue()
        img_stream_length = len(img_stream_bytes)
        img_size = numrows * numcols
        image_dtype_str, image_dtype_np = extract_image_dtype(ole, "ImageInfo")
        if image_dtype_np is not None:
            imgdata = np.frombuffer(img_stream_bytes, dtype=image_dtype_np)
        else:
            logging.error("Image could not be extracted using expected dtype '%s'", image_dtype_str)
            imgdata = fallback_image_extractor(img_stream_bytes, img_stream_length, img_size)
        imgdata.shape = (numrows, numcols)
        return imgdata
    return np.zeros([])


def extract_image_dims(ole):
    return [read_imageinfo_as_int(ole, "ImageHeight"),
            read_imageinfo_as_int(ole, "ImageWidth")]


def extract_ref_dims(ole):
    if ole.exists("ReferenceData/ImageInfo/ImageHeight"):
        return [read_imageinfo_as_int(ole, "ImageHeight", ref_data=True),
                read_imageinfo_as_int(ole, "ImageWidth", ref_data=True)]
    return extract_image_dims(ole)


def extract_number_of_images(ole):
    return read_imageinfo_as_int(ole, "NoOfImages")


def extract_tilt_angles(ole):
    return read_stream(ole, "ImageInfo/Angles", "f")


def extract_x_coords(ole):
    return np.asarray(read_stream(ole, "ImageInfo/XPosition", "f"))


def extract_y_coords(ole):
    return np.asarray(read_stream(ole, "ImageInfo/YPosition", "f"))


def extract_z_coords(ole):
    return np.asarray(read_stream(ole, "ImageInfo/ZPosition", "f"))


def extract_exposure_time(ole):
    if ole.exists('ImageInfo/ExpTimes'):
        # Returns the exposure of the image at the closest angle to 0 degrees:
        exposures = read_stream(ole, "ImageInfo/ExpTimes", "f")
        absolute_angles = np.array([abs(angle) for angle in extract_tilt_angles(ole)])
        min_angle = absolute_angles.argmin()
        return exposures[min_angle]
    elif ole.exists("ImageInfo/ExpTime"):
        return read_stream(ole, "ImageInfo/ExpTime", dtype="f")[0]
    raise Exception("No exposure time available in ole file.")


def extract_multiple_exposure_times(ole):
    if ole.exists('ImageInfo/ExpTimes'):
        # Returns the exposure of the image at the closest angle to 0 degrees:
        exposures = read_stream(ole, "ImageInfo/ExpTimes", "f")
        return exposures
    elif ole.exists("ImageInfo/ExpTime"):
        return read_stream(ole, "ImageInfo/ExpTime", dtype="f")
    raise Exception("No exposure time available in ole file.")


def extract_pixel_size(ole):
    return read_stream(ole, 'ImageInfo/PixelSize', "f")[0]


def extract_all_images(ole):
    num_rows, num_columns = extract_image_dims(ole)
    if ole.exists("ImageInfo/ExpTimes"):
        images_taken = read_imageinfo_as_int(ole, "ImagesTaken")
    elif ole.exists("ImageInfo/ExpTime"):
        images_taken = 1
    if (num_rows * num_columns > 0):
        # Iterates through images until the number of images taken
        # lambda check has been left in in case stream is wrong
        images = (extract_single_image(ole, i, num_rows, num_columns) for i in range(1, images_taken + 1))
        return np.asarray(tuple(takewhile(lambda image: image.size > 1, images)))
    raise AttributeError("No image dimensions found")


def extract_first_image(ole):
    return extract_all_images(ole)[0]


def extract_xray_magnification(ole):
    return read_stream(ole, "ImageInfo/XrayMagnification", "f")[0]


def extract_energy(ole):
    return np.mean(extract_energies(ole))


def extract_energies(ole):
    return read_stream(ole, "ImageInfo/Energy", "f")


def extract_wavelength(ole):
    return h * c / (extract_energy(ole) * e)


def create_reference_mosaic(ole, refdata, image_rows, image_columns, mosaic_rows, mosaic_columns):
    ref_num_rows = convert_to_int(image_rows / mosaic_rows)
    ref_num_columns = convert_to_int(image_columns / mosaic_columns)
    refdata.shape = (ref_num_rows, ref_num_columns)
    return np.tile(refdata, (mosaic_rows, mosaic_columns))


def rescale_ref_exposure(ole, refdata):
    # TODO: Improve this in line with ALBA's methodolgy
    # Normalises the reference exposure
    # Assumes roughly linear response, which is a reasonable estimation
    # because exposure times are unlikely to be significantly different)
    # (if it is a tomo, it does this relative to the 0 degree image, not on a per-image basis)
    ref_exp = read_stream(ole, "ReferenceData/ExpTime", dtype="f")[0]
    im_exp = extract_exposure_time(ole)
    ref_exp_rescale = im_exp / ref_exp
    if ref_exp_rescale == 1:
        return refdata
    return refdata * ref_exp_rescale


def extract_reference_image(ole):
    # Read the reference image.
    # Reference info is stored under 'ReferenceData/...'
    num_rows, num_columns = extract_ref_dims(ole)
    ref_dtype = extract_image_dtype(ole, "ReferenceData")
    ref_stream_bytes = ole.openstream("ReferenceData/Image").getvalue()
    img_size = num_rows * num_columns

    #  In XMController 13+ the reference file is not longer always a float. Additionally, there
    #  are multiple methods to apply a reference now, so this has been kept general, rather than
    #  being dependent on the file version or software version (software version is new metadata
    #  introduced in XMController 13).
    
    # Version 10 style mosaic:
    is_mosaic = ole.exists("ImageInfo/MosiacMode") and read_imageinfo_as_int(ole, "MosiacMode") == 1
    # This MosiacMode has been removed in v13 but is kept in for backward compatibility:
    # ImageInfo/MosiacMode (genuinely how it's spelled in the file) == 1 if it is a mosaic, 0 if not.
    if is_mosaic:
        mosaic_rows = read_imageinfo_as_int(ole, "MosiacRows")
        mosaic_cols = read_imageinfo_as_int(ole, "MosiacColumns")
        mosaic_size_multiplier = mosaic_rows * mosaic_cols
    else:
        mosaic_size_multiplier = 1
    ref_dtype_str, ref_dtype_np = extract_image_dtype(ole, "ReferenceData")
    if ref_dtype_np is not None:
        refdata = np.frombuffer(ref_stream_bytes, dtype=ref_dtype_np)
    else:
        logging.error("Image could not be extracted using expected dtype '%s'", ref_dtype_str)
        ref_stream_length = len(ref_stream_bytes)
        mosaic_stream_length = ref_stream_length * mosaic_size_multiplier
        refdata = fallback_image_extractor(ref_stream_bytes, mosaic_stream_length, img_size)
    if is_mosaic:
        refdata = create_reference_mosaic(ole, refdata, num_rows, num_columns, mosaic_rows, mosaic_cols)
    refdata.shape = (num_rows, num_columns)

    return rescale_ref_exposure(ole, refdata)


def convert_to_int(value):
    if int(value) == float(value):
        return int(value)
    raise ValueError(f"Value '{value}' cannot be converted to an integer")


def fallback_image_extractor(stream_bytes, stream_length, image_size):        
        if stream_length == image_size * 2:
            dtype = np.uint16
        elif stream_length == image_size * 4:
            dtype = np.float32
        else:
            logging.error("Unexpected data type with %g bytes per pixel", stream_length / image_size)
            raise TypeError("Reference is stored as unexpected type. Expecting uint16 or float32.")
        return np.frombuffer(stream_bytes, dtype=dtype)


def get_axis_dict(ole):
    """
    Gets a dictionary of all Xradia axis IDs found within the file, with a tuple of (axis name, unit) for each.
    If this fails, it returns a dictionary with keys 1-30 and a tuple of (None, None) for axis names and units.

    Args:
        ole: ole file object

    Returns:
        dict: a dictionary with the Xradia axis IDs (ints) as keys and a tuple of (axis name, unit) as the value.
    """
    ids_ = get_axis_ids(ole)
    names = get_axis_names(ole)
    units = get_all_units(ole)
    
    if (len(ids_) > 0 and len(ids_) == len(names) and len(names) == len(units)):
        return dict(zip(
            get_axis_ids(ole), zip(get_axis_names(ole), get_all_units(ole))
                ))
    else:
        num_axes = 30
        return dict(zip(
            range(1, num_axes + 1), [(None, None)] * num_axes
            ))


def get_all_units(ole):
    return axis_string_helper(ole, "AxisUnits")


def get_axis_names(ole):
    return axis_string_helper(ole, "AxisNames")


def axis_string_helper(ole, key_part):
    key1 = f"PositionInfo/{key_part}"
    key2 = f"AcquisitionSettings/{key_part}"
    if ole.exists(key1):
        stream_bytes = ole.openstream(key1).getvalue()
        try:
            return [item.decode('utf8') for item in stream_bytes.split(b'\x00') if item]
        except UnicodeDecodeError:
            return [item.decode('iso-8859-1') for item in stream_bytes.split(b'\x00') if item]
    elif ole.exists(key2):
        stream_bytes = ole.openstream(key2).getvalue()
        try:
            return [item.decode('utf8') for item in stream_bytes.split(b'\x00') if item]
        except UnicodeDecodeError:
            return [item.decode('iso-8859-1') for item in stream_bytes.split(b'\x00') if item]
    logging.error("Keys %s and %s do not exist", key1, key2)
    return []


def get_axis_ids(ole):
    if ole.exists("PositionInfo/IndexToXradiaID"):
        return read_stream(ole, "PositionInfo/IndexToXradiaID", "i")
    elif ole.exists("AcquisitionSettings/IndexToXradiaID"):
        return read_stream(ole, "AcquisitionSettings/IndexToXradiaID", "i")
    logging.error("Failed to read axis IDs")
    return []
