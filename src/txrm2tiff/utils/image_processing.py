import logging
from typing import List
import numpy as np
from numpy.typing import DTypeLike

from .functions import convert_to_int


def dynamic_despeckle_and_average_series(
    images: np.ndarray, average: bool = True
) -> np.ndarray:
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
        logging.error("Despeckle averaging requires a minimum of 3 images")
        return None
    # Takes a list of XradiaData.FloatArray as input
    # outputs a 2D ndarray
    block = images.astype(np.float32).reshape((nimages, height * width))
    # Sort pixels stack-wise
    vals = np.sort(block, axis=0)
    # Set number of sorted frames to split off
    maligned_idxs = [nimages - 1]
    if nimages >= 9:
        maligned_idxs.extend([nimages - 2, nimages - 3, 0])
        good_vals = vals[1:-3]
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


def normalise_to_datatype(
    array: np.ndarray, datatype: DTypeLike, clip: bool = False
) -> np.ndarray:
    logging.debug("Re-scaling array to %s", datatype)
    array = array.astype(np.float64)  # Convert to float for rescaling

    # Clip outliers before normalisation to avoid any speckles
    # that can lead to the normalised image to be dark:
    if clip:
        num_std = 3.0
        logging.info("Clipping outliers (>%g std from mean)", num_std)
        minimum = 0
        maximum = np.mean(array) + num_std * np.std(array)
    else:
        minimum = array.min()
        maximum = array.max()

    return rescale_image(
        array,
        np.iinfo(datatype).min,
        np.iinfo(datatype).max,
        previous_minimum=minimum,
        previous_maximum=maximum,
    )


def rescale_image(
    array, minimum, maximum, previous_minimum=None, previous_maximum=None
):
    if previous_minimum is None:
        previous_minimum = np.min(array)
    if previous_maximum is None:
        previous_maximum = np.max(array)

    return np.interp(
        np.clip(array, a_min=previous_minimum, a_max=previous_maximum),
        (previous_minimum, previous_maximum),
        (minimum, maximum),
    )


def cast_to_dtype(image: np.ndarray, data_type: DTypeLike) -> np.ndarray:
    try:
        dtype = np.dtype(data_type)
        if dtype is image.dtype:
            logging.debug("Image is already %s", dtype)
        else:
            if np.issubdtype(dtype, np.integer):
                dtype_info = np.iinfo(dtype)
            elif np.issubdtype(dtype, np.floating):
                dtype_info = np.finfo(dtype)
            else:
                raise TypeError(f"Cannot cast to invalid data type {data_type}")
            # Round min/max to avoid this warning when the issue is just going to be rounded away.
            img_min, img_max = image.min(), image.max()
            dtype_min, dtype_max = dtype_info.min, dtype_info.max
            if dtype_min > img_min:
                logging.warning(
                    "Image min %f below %s minimum of %i, values below this will be cut off",
                    img_min,
                    dtype,
                    dtype_min,
                )
            else:
                dtype_min = None
            if dtype_max < img_max:
                logging.warning(
                    "Image max %f above %s maximum of %i, values above this will be cut off",
                    img_max,
                    dtype,
                    dtype_max,
                )
            else:
                dtype_max = None

            if dtype_min is not None or dtype_max is not None:
                np.clip(image, dtype_min, dtype_max, out=image)

            if np.issubdtype(dtype, np.integer) and np.issubdtype(
                image.dtype, np.floating
            ):
                image = np.around(image, decimals=0)

            image = image.astype(dtype, copy=False)
            logging.info("Image has been cast to %s", data_type)
    except TypeError:
        logging.warning(
            "Invalid data type given: %s aka %s. Image will remain as %s.",
            data_type,
            dtype,
            image.dtype,
        )
    except Exception:
        logging.error(
            "An error occurred casting image from %s to %s", image.dtype, data_type
        )
        raise
    return image


def tile_image_data_to_mosaic(
    refdata: np.ndarray,
    image_rows: int,
    image_columns: int,
    mosaic_rows: int,
    mosaic_columns: int,
) -> np.ndarray:
    ref_num_rows = convert_to_int(image_rows / mosaic_rows)
    ref_num_columns = convert_to_int(image_columns / mosaic_columns)
    refdata.shape = (ref_num_rows, ref_num_columns)
    return np.tile(refdata, (mosaic_rows, mosaic_columns))


def stitch_images(images: np.ndarray, mosaic_dims: List[int]) -> np.ndarray:
    """
    Stitches images into a mosaic stored as a 2D array.

    Mosaic dims should be X, Y.
    """
    # Convert to Y X for numpy:
    slow_np_axis = 0
    fast_np_axis = 1

    num_images, y, x = images.shape
    expected_num_images = mosaic_dims[0] * mosaic_dims[1]

    if expected_num_images > num_images:
        images.resize((expected_num_images, y, x), refcheck=False)

    fast_stacked_list = []
    for i in range(mosaic_dims[slow_np_axis]):
        idx_start = mosaic_dims[fast_np_axis] * i
        idx_end = mosaic_dims[fast_np_axis] * (i + 1)
        logging.debug(
            "Stitching mosaic images %i to %i along axis %s",
            idx_start,
            (idx_end - 1),
            ("x", "y")[fast_np_axis],
        )
        fast_stacked_list.append(
            np.concatenate(images[idx_start:idx_end], axis=fast_np_axis)
        )
    # Must be output in a 3D numpy array (with axis 0 as z)
    final_image = np.concatenate(fast_stacked_list, axis=slow_np_axis)[np.newaxis, :, :]
    logging.info("Mosaic stitched into image with shape: %s", final_image.shape)
    return final_image
