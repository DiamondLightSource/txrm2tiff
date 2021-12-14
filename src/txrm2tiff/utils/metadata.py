import logging
import numpy as np
from oxdls import OMEXML, DO_XYZCT, PT_UINT16, PT_UINT16, PT_FLOAT, PT_DOUBLE

from ..txrm import abstract

dtype_dict = {"uint16": PT_UINT16, "float32": PT_FLOAT, "float64": PT_DOUBLE}




def create_ome_metadata(txrm: abstract.AbstractTxrm, filename: str = None) -> OMEXML:
    # Get image shape
    # number of frames (T in tilt series), Y, X:
    shape = txrm.output_shape

    # Get metadata variables from ole file:
    exposures = txrm.exposures.copy()

    pixel_size = txrm.image_info.get("PixelSize", (0,))[0] * 1.0e3  # micron to nm

    x_positions = [
        coord * 1.0e3 for coord in txrm.image_info["XPosition"]
    ]  # micron to nm
    y_positions = [
        coord * 1.0e3 for coord in txrm.image_info["YPosition"]
    ]  # micron to nm
    z_positions = [
        coord * 1.0e3 for coord in txrm.image_info["ZPosition"]
    ]  # micron to nm

    ox = OMEXML()

    image = ox.image()
    image.set_ID("Image:0")
    image.set_AcquisitionDate(txrm.datetimes[0].isoformat())  # formatted as: "yyyy-mm-ddThh:mm:ss"
    if filename is not None:
        image.set_Name(filename)

    pixels = image.Pixels
    pixels.DimensionOrder = DO_XYZCT
    pixels.set_ID("Pixels:0")

    pixels.set_SizeX(shape[2])
    pixels.set_SizeY(shape[1])
    pixels.set_SizeC(1)
    pixels.set_SizeZ(shape[0])

    pixels.set_SizeT(1)

    # Physical size of a pixel
    pixels.set_PhysicalSizeX(pixel_size)
    pixels.set_PhysicalSizeXUnit("nm")
    pixels.set_PhysicalSizeY(pixel_size)
    pixels.set_PhysicalSizeYUnit("nm")
    pixels.set_PhysicalSizeZ(1)
    pixels.set_PhysicalSizeZUnit("reference frame")

    pixels.set_tiffdata_count(shape[0])  # tiffdata must be created before planes
    pixels.set_plane_count(shape[0])

    channel = pixels.Channel(0)
    channel.set_ID("Channel:0:0")
    channel.set_Name("C:0")

    if txrm.is_mosaic:
        mosaic_columns, mosaic_rows = txrm.mosaic_dims
        # Calculates:
        # - Mean exposure, throwing away any invalid 0 values
        # - The centre of the stitched mosaic image (as opposed to the centre of a single tile)
        # Both should be returned as a list to reduce changes to the next section.
        valid_idxs = np.nonzero(exposures)[0]
        exposures = [np.mean(np.asarray(exposures)[valid_idxs])]
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
        x_positions = [
            x_positions[0]
            + (1.0 - 1.0 / mosaic_columns) * (pixel_size * shape[2] / 2.0)
        ]
        y_positions = [
            y_positions[0] + (1.0 - 1.0 / mosaic_rows) * (pixel_size * shape[1] / 2.0)
        ]
        z_positions = [np.mean(np.asarray(z_positions)[valid_idxs])]  # Average Z for a stitched mosaic
        # # NOTE: the number of mosaic rows & columns and the pixel size are all written before acquisition but
        # the xy positions are written during, so only the first frame can be relied upon to have an xy
        # position.

    # Run checks to make sure the value lists are long enough
    exp_len_diff = shape[0] - len(exposures)
    if exp_len_diff > 0:
        logging.error(
            "Not enough exposure values for each plane (%i vs %i). Adding zeros to the later planes.",
            len(exposures),
            shape[0],
        )
        for _ in range(exp_len_diff):
            exposures.append(0)

    x_len_diff = shape[0] - len(x_positions)
    if x_len_diff > 0:
        logging.error(
            "Not enough x values for each plane (%i vs %i). Adding zeros to the later planes.",
            len(x_positions),
            shape[0],
        )
        for _ in range(x_len_diff):
            x_positions.append(0)

    y_len_diff = shape[0] - len(y_positions)
    if y_len_diff > 0:
        logging.error(
            "Not enough y values for each plane (%i vs %i). Adding zeros to the later planes.",
            len(y_positions),
            shape[0],
        )
        for _ in range(y_len_diff):
            y_positions.append(0)

    # Add plane/tiffdata for each plane in the stack
    for count in range(shape[0]):
        tiffdata = pixels.tiffdata(count)
        tiffdata.set_FirstC(0)
        tiffdata.set_FirstZ(count)
        tiffdata.set_FirstT(0)
        tiffdata.set_IFD(count)
        tiffdata.set_plane_count(1)

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
        plane.set_PositionZ(z_positions[count])

    return ox
