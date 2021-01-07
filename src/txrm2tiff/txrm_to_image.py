# -*- coding: utf-8 -*-

from xml.etree import ElementTree
import datetime
import logging
from pathlib import Path

from olefile import OleFileIO, isOleFile
import numpy as np
import tifffile as tf
from oxdls import OMEXML

from . import txrm_wrapper


dtype_dict = {
    'uint16': np.uint16,
    'float32': np.float32,
    'float64': np.float64
}

def _dynamic_despeckle_and_average_series(images, average=True):
    """
    Despeckle and average a series of images (requires min 3 images)
    This uses the same or similar logic to Xradia/Zeiss' API.

    Args:
        images (list): List of 2D numpy arrays
        average (bool, optional): Defaults to True.

    Returns:
        numpy array: single image (2D array) containing despeckle and averaged data
    """
    nimages = len(images)
    if nimages < 3:
        logging.error('Despeckle averaging requires a minimum of 3 images')
        return None
    # Takes a list of XradiaData.FloatArray as input
    # outputs a 2D ndarray
    height, width = images[0].shape
    block = np.asarray([image.flatten() for image in images], dtype=np.float32)
    # Sort pixels stack-wise
    vals = np.sort(block, axis=0)
    # Set number of sorted frames to split off
    maligned_idxs = [nimages-1]
    if nimages >= 9:
        maligned_idxs.extend([nimages-2, nimages-3, 0])
        good_vals = vals[1: -3]
    else:
        good_vals = vals[:-1]
    maligned = vals[maligned_idxs, :]
    # Calculate mean and sigma of 'good' pixels
    mean = np.mean(good_vals, axis=0, keepdims=True)
    sigma = np.std(good_vals, axis=0, ddof=1, keepdims=True)
    # Create boolean mask for cutting out 'bad' pixels
    bad_pixels = np.fabs(maligned - mean) > 2.8 * sigma
    # Set bad pixels to nan and apply to vals
    maligned[bad_pixels] = np.nan
    vals[maligned_idxs] = maligned
    # Get stack-wise mean (ignoring invalid nans)
    output_image = np.nanmean(vals, axis=0, dtype=np.float32)
    if not average:
        output_image *= nimages
    return output_image.reshape(height, width)


def _apply_reference(images, reference):
    floated_and_referenced = [((image * 100.) / reference) for image in images]
    referenced_image = []
    for image in floated_and_referenced:
        if np.isnan(image).any() or np.isinf(image).any():
            logging.warning("Potential dead pixels found. "
                            "NaN was output for at least one pixel in the referenced image.")
            # Replace any infinite pixels (nan or inf) with 0:
            invalid = np.where(np.logical_not(np.isfinite(image)))
            image[invalid] = 0
            # convert to float32 as divide returns float64
            image = image.astype(np.float32)
        referenced_image.append(np.around(image))
    return referenced_image


def _get_reference(ole, txrm_name, custom_reference, ignore_reference):
    if custom_reference is not None:
        logging.info("%s is being processed with file %s as a reference.", txrm_name, custom_reference.name)
        reference_path = str(custom_reference)
        try:
            if isOleFile(reference_path):
                with OleFileIO(reference_path) as ref_ole:
                    references = txrm_wrapper.extract_all_images(ref_ole)  # should be float for averaging & dividing
            elif ".tif" in reference_path:
                with tf.TiffFile(reference_path) as tif:
                    references = [page for page in tif.pages[:]]
            else:
                msg = f"Unable to open file '{reference_path}'. Only tif/tiff or xrm/txrm files are supported for custom references."
                logging.error(msg)
                raise IOError(msg)
        except:
            logging.error("Error occurred reading custom reference", exc_info=True)
            raise
        if len(references) > 1:
            # if reference file is an image stack take median of the images
            return _dynamic_despeckle_and_average_series(references)
        return references[0]

    elif ole.exists("ReferenceData/Image") and not ignore_reference:
        logging.info("internal reference will be applied to %s", txrm_name)
        return txrm_wrapper.extract_reference_image(ole)

    logging.debug("%s is being processed without a reference.", txrm_name)
    return None

def _stitch_images(img_list, mosaic_xy_shape, fast_axis):
    slow_axis = 1 - fast_axis
    logging.debug("Fast axis: %i", fast_axis)
    
    num_images = len(img_list)
    expected_num_images = mosaic_xy_shape[0] * mosaic_xy_shape[1]
    if expected_num_images > num_images:  # Pad with zeros if mosaic interrupted
        img_list.extend([np.zeros((img_list[0].shape), dtype=img_list[0].dtype)] * (expected_num_images - num_images))
    
    fast_stacked_list = []
    for i in range(mosaic_xy_shape[slow_axis]):
        idx_start = mosaic_xy_shape[fast_axis] * i
        idx_end = mosaic_xy_shape[fast_axis] * (i + 1)
        logging.debug("Stacking images %i to %i along axis %i", idx_start, (idx_end - 1), fast_axis)
        fast_stacked_list.append(np.concatenate(img_list[idx_start: idx_end], axis=fast_axis))
    # Must be output in a list for iterative tiff saving
    final_image = np.concatenate(fast_stacked_list, axis=slow_axis)
    logging.info("Mosaic stitched into image with shape: %s", final_image.shape)
    return [final_image]


def manual_save(tiff_file, image, data_type=None, metadata=None):
    tiff_path = Path(tiff_file)

    if data_type is not None:
        try:
            dtype = np.dtype(data_type).type
            image = [frame.astype(dtype) for frame in image]
        except Exception as e:
            logging.error("Invalid data type given: %s aka %s. Saving with default data type.", data_type, dtype)
    else:
        logging.error("No data type specified. Saving with default data type.")

    if metadata is not None:
        meta_img = metadata.image()
        meta_img.Pixels.set_PixelType(str(image[0].dtype))
        meta_img.set_Name(tiff_path.name)
        metadata = metadata.to_xml().encode()

    tiff_dir = tiff_path.resolve().parent
    tiff_dir.mkdir(parents=True, exist_ok=True)
    num_frames = len(image)
    logging.info("Saving image as %s with %i frames", tiff_path.name, num_frames)

    with tf.TiffWriter(str(tiff_path)) as tif:
        tif.save(image[0], photometric='minisblack', description=metadata, metadata={'axes':'XYZCT'})
        for i in range(1, num_frames):
            tif.save(image[i], photometric='minisblack', metadata={'axes':'XYZCT'})

def create_ome_metadata(ole, image_list, filename=None):
    # Get image dimensions
    # X, Y, number of frames (T in tilt series):
    dimensions = (*image_list[0].shape, len(image_list))
    str_dtype = str(image_list[0].dtype)

    # Get metadata variables from ole file:
    exposures = txrm_wrapper.extract_multiple_exposure_times(ole)

    pixel_size = txrm_wrapper.extract_pixel_size(ole) * 1.e3  # micron to nm

    physical_img_sizes = []
    physical_img_sizes.append(pixel_size * dimensions[0])
    physical_img_sizes.append(pixel_size * dimensions[1])

    x_positions = [coord * 1.e3 for coord in txrm_wrapper.extract_x_coords(ole)]  # micron to nm
    y_positions = [coord * 1.e3 for coord in txrm_wrapper.extract_y_coords(ole)]  # micron to nm

    date_time = datetime.datetime.now().isoformat()  # formatted as: "yyyy-mm-ddThh:mm:ss"
    ox = OMEXML()

    image = ox.image()
    image.set_ID("0")
    image.set_AcquisitionDate(date_time)
    if filename:
        image.set_Name(filename)

    pixels = image.Pixels
    pixels.set_DimensionOrder("XYTZC")
    pixels.set_ID("0")
    pixels.set_PixelType(str_dtype)
    pixels.set_SizeX(dimensions[0])
    pixels.set_SizeY(dimensions[1])
    pixels.set_SizeT(dimensions[2])
    pixels.set_SizeZ(1)
    pixels.set_SizeC(1)
    pixels.set_PhysicalSizeX(physical_img_sizes[0])
    pixels.set_PhysicalSizeXUnit("nm")
    pixels.set_PhysicalSizeY(physical_img_sizes[1])
    pixels.set_PhysicalSizeYUnit("nm")

    pixels.set_plane_count(dimensions[2])
    pixels.set_tiffdata_count(dimensions[2])

    channel = pixels.Channel(0)
    channel.set_ID("Channel:0:0")
    channel.set_Name("C:0")

    if (ole.exists("ImageInfo/MosiacRows") and ole.exists("ImageInfo/MosiacColumns")):
        mosaic_rows = txrm_wrapper.read_imageinfo_as_int(ole, "MosiacRows")
        mosaic_columns = txrm_wrapper.read_imageinfo_as_int(ole, "MosiacColumns")
        # Checks whether the output is a single frame and a mosaic
        # (mosaics should be stitched, and therefore one layer, by now):
        if (mosaic_rows > 0 and mosaic_columns > 0 and dimensions[2] == 1):
            # Calculates:
            # - Mean exposure, throwing away any invalid 0 values
            # - The centre of the stitched mosaic image (as opposed to the centre of a single tile)
            # Both should be returned as a list to reduce changes to the next section.
            exposures = [np.mean([exp for exp in exposures if exp != 0])]
            # The mosaic centre is found by taking the first x & y positions (centre of the first tile,
            # which is the bottom-left in the mosaic), taking away the distance between this and the
            # bottom-left corner, then adding the distance to the centre of the mosaic (calculated using pixel size).
            #
            # More verbosely:
            # The physical size of the stitched mosaic is divided by the rows/columns (columns for x, rows for y).
            # This finds the physical size of a single tile. This is then halved, finding in the physical
            # distance (x, y) from between a corner and the centre of a tile. Then this distance is taken from the
            # (x, y) stage coordinates of the first tile to get the stage coordinates of the bottom-left of the mosaic.
            # Half the physical size of the stitched mosaic is added to this, resulting in the in stage coordinates
            # of the mosaic centre.
            x_positions = [x_positions[0] + (1. - 1. / mosaic_columns) * (physical_img_sizes[0] / 2.)]
            y_positions = [y_positions[0] + (1. - 1. / mosaic_rows) * (physical_img_sizes[1] / 2.)]
            # # NOTE: the number of mosaic rows & columns and the pixel size are all written before acquisition but
            # the xy positions are written during, so only the first frame can be relied upon to have an xy
            # position.

    # Run checks to make sure the value lists are long enough
    exp_len_diff = dimensions[2] - len(exposures)
    if exp_len_diff > 0:
        logging.error("Not enough exposure values for each plane (%i vs %i). Adding zeros to the later planes.",
                        len(exposures), dimensions[2])
        for _ in range(exp_len_diff):
            exposures.append(0)

    x_len_diff = dimensions[2] - len(x_positions)
    if x_len_diff > 0:
        logging.error("Not enough x values for each plane (%i vs %i). Adding zeros to the later planes.",
                        len(x_positions), dimensions[2])
        for _ in range(x_len_diff):
            x_positions.append(0)

    y_len_diff = dimensions[2] - len(y_positions)
    if y_len_diff > 0:
        logging.error("Not enough y values for each plane (%i vs %i). Adding zeros to the later planes.",
                        len(y_positions), dimensions[2])
        for _ in range(y_len_diff):
            y_positions.append(0)

    # Add plane/tiffdata for each plane in the stack
    for count in range(dimensions[2]):
        plane = pixels.Plane(count)
        plane.set_ExposureTime(exposures[count])
        plane.set_TheZ(count)
        plane.set_TheC(0)
        plane.set_TheT(0)

        plane.set_PositionXUnit("nm")
        plane.set_PositionYUnit("nm")
        plane.set_PositionZUnit("reference frame")
        plane.set_PositionX(x_positions[count])
        plane.set_PositionY(y_positions[count])
        plane.set_PositionZ(count)

        tiffdata = pixels.tiffdata(count)
        tiffdata.set_FirstC(0)
        tiffdata.set_FirstZ(count)
        tiffdata.set_FirstT(0)
        tiffdata.set_IFD(count)
        tiffdata.set_plane_count(1)

    return ox

class TxrmToImage:

    def __init__(self):
        self.image_output = None
        self.ome_metadata = None

    def convert(self, txrm_file, custom_reference=None, ignore_reference=False):
        with OleFileIO(str(txrm_file)) as ole:
            images = txrm_wrapper.extract_all_images(ole)
            reference = _get_reference(ole, txrm_file.name, custom_reference, ignore_reference)
            if reference is not None:
                self.image_output = _apply_reference(images, reference)
            else:
                self.image_output = [image for image in np.around(images)]
            if (len(self.image_output) > 1
                    and ole.exists("ImageInfo/MosiacRows")
                    and ole.exists("ImageInfo/MosiacColumns")):
                mosaic_rows = txrm_wrapper.read_imageinfo_as_int(ole, "MosiacRows")
                mosaic_cols = txrm_wrapper.read_imageinfo_as_int(ole, "MosiacColumns")
                if mosaic_rows != 0 and mosaic_cols != 0:
                    # Version 13 style mosaic:
                    self.image_output = _stitch_images(self.image_output, (mosaic_cols, mosaic_rows), 1)
            # Create metadata
            self.ome_metadata = create_ome_metadata(ole, self.image_output)

    def get_image_and_metadata(self):
        if self.image_output is None:
            logging.warning("Image has not been converted, returning (None, None)")
        return self.image_output, self.ome_metadata


    def save(self, tiff_file, data_type=None):
        if (self.image_output is not None) and (self.ome_metadata is not None):
            manual_save(tiff_file, self.image_output, data_type, self.ome_metadata)
        else:
            logging.error("Nothing to save! Please convert the image first")
            raise IOError("Nothing to save! Please convert the image first")
