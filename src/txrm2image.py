# -*- coding: utf-8 -*-

from xml.etree import ElementTree
import uuid
from olefile import OleFileIO
import datetime
import numpy as np
import tifffile as tf
import logging

from txrm_wrapper import TxrmWrapper
from omexml.omexml import OMEXML


class TxrmToTiff:

    def __init__(self, datatype="uint16"):
        self.txrm_extractor = TxrmWrapper()
        self.datatype = datatype

    def apply_reference(self, images, reference):
        floated_and_referenced = [(image * 100.) / reference for image in images]
        referenced_image = []
        for image in floated_and_referenced:
            if np.isnan(image).any() or np.isinf(image).any():
                logging.warn("Potential dead pixels found. "
                             "NaN was output for at least one pixel in the referenced image.")
            referenced_image.append(np.around(image).astype("uint16"))

        return referenced_image

    def convert(self, txrm_file, tiff_file, custom_reference, ignore_reference):
        ole = OleFileIO(str(txrm_file))
        images = self.txrm_extractor.extract_all_images(ole)
        if custom_reference is not None:
            logging.debug("%s is being referenced using %s and processed.", txrm_file.name, custom_reference.name)
            ref_ole = OleFileIO(str(txrm_file))
            references = self.txrm_extractor.extract_all_images(ref_ole)  # should be float for averaging & dividing
            if len(references) > 1:
                # take median across z-axes (i.e. over time) if an image stack
                reference = np.median(np.asarray(references), axis=0)
            else:
                reference = references[0]
            image_output = self.apply_reference(images, reference)
        elif ole.exists("ReferenceData/Image") and not ignore_reference:
            logging.debug("%s is being referenced and processed.", txrm_file.name)
            reference = self.txrm_extractor.extract_reference_image(ole)
            image_output = self.apply_reference(images, reference)
        else:
            logging.debug("%s is being processed without a reference.", txrm_file.name)
            image_output = [image for image in np.around(images).astype(self.datatype)]
        if (len(image_output) > 1
                and ole.exists("ImageInfo/MosiacRows")
                and ole.exists("ImageInfo/MosiacColumns")):
            mosaic_rows = self.txrm_extractor.read_imageinfo_as_int(ole, "MosiacRows")
            mosaic_cols = self.txrm_extractor.read_imageinfo_as_int(ole, "MosiacColumns")
            if mosaic_rows != 0 and mosaic_cols != 0:
                image_output = self.stitch_images(image_output, (mosaic_cols, mosaic_rows), 1)

        # Get image dimensions
        x, y = image_output[0].shape
        z = len(image_output)
        dimensions = (x, y, z)  # X, Y, Z

        logging.info("Saving image as %s with %i frames", txrm_file.name, z)

        # Create metadata
        ome_metadata = self.create_ome_metadata(str(tiff_file.name), ole, dimensions)

        tiff_dir = tiff_file.resolve().parent
        if not tiff_dir.is_dir():
            tiff_dir.mkdir(parents=True)
        with tf.TiffWriter(str(tiff_file)) as tif:
            tif.save(image_output[0], photometric='minisblack', description=ome_metadata, metadata={'axes':'XYZ'})
            for i in range(1, z):
                tif.save(image_output[i], photometric='minisblack', metadata={'axes':'XYZ'})
            tif.close()
        ole.close()

    @staticmethod
    def stitch_images(img_list, mosaic_xy_shape, fast_axis):
        slow_axis = 1 - fast_axis
        logging.debug("Fast axis: %s", fast_axis)
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

    def create_ome_metadata(self, image_name, ole, dimensions):

        # Get metadata variables from ole file:
        exposures = self.txrm_extractor.extract_multiple_exposure_times(ole)

        pixel_size = self.txrm_extractor.extract_pixel_size(ole) * 1.e3  # micron to nm

        physical_img_sizes = []
        physical_img_sizes.append(pixel_size * dimensions[0])
        physical_img_sizes.append(pixel_size * dimensions[1])

        x_positions = self.txrm_extractor.extract_x_coords(ole) * 1.e3  # micron to nm
        y_positions = self.txrm_extractor.extract_y_coords(ole) * 1.e3  # micron to nm

        date_time = datetime.datetime.now().isoformat()  # formatted as: "yyyy-mm-ddThh:mm:ss"
        ox = OMEXML()

        image = ox.image()
        image.set_Name(image_name)
        image.set_ID("0")
        image.set_AcquisitionDate(date_time)

        pixels = image.Pixels

        pixels.set_DimensionOrder("XYZ")
        pixels.set_ID("0")
        pixels.set_PixelType(self.datatype)
        pixels.set_SizeX(dimensions[0])
        pixels.set_SizeY(dimensions[1])
        pixels.set_SizeZ(dimensions[2])
        pixels.set_SizeC(1)
        pixels.set_SizeT(1)
        pixels.set_PhysicalSizeX(physical_img_sizes[0])
        pixels.set_PhysicalSizeXUnit("nm")
        pixels.set_PhysicalSizeY(physical_img_sizes[1])
        pixels.set_PhysicalSizeYUnit("nm")
        pixels.set_PhysicalSizeZ(1)  # Z doesn't have corresponding data
        pixels.set_PhysicalSizeZUnit("reference frame")

        pixels.set_plane_count(dimensions[2])
        pixels.set_tiffdata_count(dimensions[2])

        channel = pixels.Channel(0)
        channel.set_ID("Channel:0:0")
        channel.set_Name("C:0")

        if (ole.exists("ImageInfo/MosiacRows") and ole.exists("ImageInfo/MosiacColumns")):
            mosaic_rows = self.txrm_extractor.read_imageinfo_as_int(ole, "MosiacRows")
            mosaic_columns = self.txrm_extractor.read_imageinfo_as_int(ole, "MosiacColumns")
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

            tiffdata = pixels.TiffData(count)
            tiffdata.set_FirstC(0)
            tiffdata.set_FirstZ(count)
            tiffdata.set_FirstT(0)
            tiffdata.set_IFD(count)
            tiffdata.set_PlaneCount(1)

        return ox.to_xml().encode()
