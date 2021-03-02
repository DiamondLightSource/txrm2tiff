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
from .annotator import Annotator

from .__init__ import __version__

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
    nimages, height, width = images.shape
    if nimages < 3:
        logging.error('Despeckle averaging requires a minimum of 3 images')
        return None
    # Takes a list of XradiaData.FloatArray as input
    # outputs a 2D ndarray
    block = images.astype(np.float32).reshape((nimages, height * width))
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
    floated_and_referenced = np.asarray(images, dtype=np.float32) * 100. / reference
    if np.isnan(floated_and_referenced).any() or np.isinf(floated_and_referenced).any():
        logging.warning("Potential dead pixels found. "
                        "NaN was output for at least one pixel in the referenced image.")
        # Replace any infinite pixels (nan or inf) with 0:
        _conditional_replace(floated_and_referenced, 0, lambda x: ~np.isfinite(x))
    return floated_and_referenced.astype(np.float32)


def _conditional_replace(array, replacement, condition_func):
    array[condition_func(array)] = replacement


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
                    references = np.asarray(tif.pages[:])
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
        logging.info("Internal reference will be applied to %s", txrm_name)
        return txrm_wrapper.extract_reference_image(ole)

    logging.debug("%s is being processed without a reference.", txrm_name)
    return None

def _stitch_images(images, mosaic_xy_shape, fast_axis):
    slow_axis = 1 - fast_axis
    axis_names = ("x", "y")
    logging.debug("Fast axis: %s", axis_names[fast_axis])
    num_images, y, x = images.shape
    expected_num_images = mosaic_xy_shape[0] * mosaic_xy_shape[1]
    
    if expected_num_images > num_images:
        images.resize((expected_num_images, y, x), refcheck=False)

    fast_stacked_list = []
    for i in range(mosaic_xy_shape[slow_axis]):
        idx_start = mosaic_xy_shape[fast_axis] * i
        idx_end = mosaic_xy_shape[fast_axis] * (i + 1)
        logging.debug("Stacking images %i to %i along axis %s", idx_start, (idx_end - 1), axis_names[fast_axis])
        fast_stacked_list.append(np.concatenate(images[idx_start: idx_end], axis=fast_axis))
    # Must be output in a 3D numpy array (with axis 0 as z)
    final_image = np.concatenate(fast_stacked_list, axis=slow_axis)[np.newaxis, :, :]
    logging.info("Mosaic stitched into image with shape: %s", final_image.shape)
    return final_image

def manual_save(tiff_file, image, data_type=None, metadata=None):
    tiff_path = Path(tiff_file)
    image = np.asarray(image)
    image = _cast_to_dtype(image, data_type)
    if metadata is not None:
        meta_img = metadata.image()
        meta_img.Pixels.set_PixelType(str(image.dtype))
        meta_img.set_Name(tiff_path.stem)
        metadata = metadata.to_xml().encode()

    tiff_dir = tiff_path.resolve().parent
    tiff_dir.mkdir(parents=True, exist_ok=True)
    num_frames = len(image)
    bigtiff = image.size * image.itemsize >= np.iinfo(np.uint32).max  # Check if data bigger than 4GB TIFF limit

    logging.info("Saving image as %s with %i frames", tiff_path.name, num_frames)

    with tf.TiffWriter(str(tiff_path), bigtiff=bigtiff, ome=False) as tif:
        tif.write(
            image,
            photometric='MINISBLACK',
            description=metadata,
            metadata={'axes': 'ZYX'},
            software=f"txrm2tiff {__version__}"
            )
    
def save_colour(tiff_file, image):
    tiff_path = Path(tiff_file)
    num_frames = len(image)
    logging.info("Saving annotated image as %s with %i frames", tiff_path.name, num_frames)
    bigtiff = image.size * image.itemsize >= np.iinfo(np.uint32).max  # Check if data bigger than 4GB TIFF limit
    with tf.TiffWriter(str(tiff_path), bigtiff=bigtiff, ome=False) as tif:
        tif.write(
            image,
            photometric='RGB',
            metadata={'axes':'ZYXC'},
            software=f"txrm2tiff {__version__}"
            )


def _convert_output_path_to_annotated_path(output_path):
    """
    Args:
        output_path (pathlib.Path): output path to add "_Annotated" to
    """""
    split_name = output_path.name.split(".")
    num_parts = len(split_name)
    name_idx = num_parts
    suffix = ""
    if num_parts > 1:
        suffix = f".{split_name[-1]}"
        name_idx -= 1
        if num_parts > 2 and split_name[-2].lower() == "ome":
            name_idx -= 1
    annotated_name = ".".join([s for s in split_name[:name_idx]])
    annotated_name += f"_Annotated{suffix}"
    return output_path.parent / annotated_name


def _cast_to_dtype(image, data_type):
    if data_type is not None:
        try:
            dtype = np.dtype(data_type)
            if np.issubdtype(dtype, np.integer) and np.issubdtype(image.dtype, np.floating):
                dtype_info = np.iinfo(dtype)
                # Round min/max to avoid this warning when the issue is just going to be rounded away.
                img_min, img_max = round(image.min()), round(image.max())
                if dtype_info.min > img_min:
                    logging.warning(
                        "Image min %f below %s minimum of %i, values below this will be cut off",
                        img_min, dtype, dtype_info.min)
                    _conditional_replace(image, dtype_info.min, lambda x: x < dtype_info.min)
                if dtype_info.max < img_max:
                    logging.warning(
                        "Image max %f above %s maximum of %i, values above this will be cut off",
                        img_max, dtype, dtype_info.max)
                    _conditional_replace(image, dtype_info.max, lambda x: x > dtype_info.max)
            return np.around(image, decimals=0).astype(dtype)
        except Exception:
            logging.error("Invalid data type given: %s aka %s. Saving with default data type.", data_type, dtype)
    else:
        logging.warning("No data type specified. Saving with default data type.")
    return image


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
    pixels.set_DimensionOrder("XYZCT")
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
        self.annotator = None
        self.ome_metadata = None

    def convert(self, txrm_file, custom_reference=None, ignore_reference=False, annotate=False):
        with OleFileIO(str(txrm_file)) as ole:
            images = txrm_wrapper.extract_all_images(ole)
            reference = _get_reference(ole, txrm_file.name, custom_reference, ignore_reference)
            if reference is not None:
                self.image_output = _apply_reference(images, reference)
            else:
                self.image_output = np.around(images)
            if (len(self.image_output) > 1
                    and ole.exists("ImageInfo/MosiacRows")
                    and ole.exists("ImageInfo/MosiacColumns")):
                mosaic_rows = txrm_wrapper.read_imageinfo_as_int(ole, "MosiacRows")
                mosaic_cols = txrm_wrapper.read_imageinfo_as_int(ole, "MosiacColumns")
                if mosaic_rows != 0 and mosaic_cols != 0:
                    # Version 13 style mosaic:
                    self.image_output = _stitch_images(self.image_output, (mosaic_cols, mosaic_rows), 1)
            if annotate:
                # Extract annotations
                annotator = Annotator(self.image_output[0].shape[::-1])
                if annotator.extract_annotations(ole):  # True if any annotations were drawn
                    self.annotator = annotator
                else:
                    self.annotator = False
            # Create metadata
            self.ome_metadata = create_ome_metadata(ole, self.image_output)

    def get_image_and_metadata(self):
        if self.image_output is None:
            logging.warning("Image has not been converted, returning (None, None)")
        return self.image_output, self.ome_metadata

    def get_annotator(self):
        return self.annotator

    def get_annotated_images(self):
        if self.annotator is not None:
            return self.annotator.apply_annotations(self.image_output)
        return None

    def save(self, tiff_file, data_type=None):
        tiff_path = Path(tiff_file)
        if (self.image_output is not None) and (self.ome_metadata is not None):
            manual_save(tiff_path, self.image_output, data_type, self.ome_metadata)
            if self.annotator is None:
                pass
            elif not self.annotator:
                logging.error("No annotations or scale bar to save with %s", tiff_path)
            else:
                annotationed_images = self.annotator.apply_annotations(self.image_output)
                if annotationed_images is not None:
                    annotated_path = _convert_output_path_to_annotated_path(tiff_path)
                    save_colour(annotated_path, annotationed_images)
        else:
            logging.error("Nothing to save! Please convert the image first")
            raise IOError("Nothing to save! Please convert the image first")
