# -*- coding: utf-8 -*-

from xml.etree import ElementTree
import datetime
import logging
from pathlib import Path

from olefile import OleFileIO, isOleFile
import numpy as np
import tifffile as tf
from oxdls import OMEXML

from .txrm_wrapper import TxrmWrapper


class TxrmToImage:

    def __init__(self, dtype="uint16"):
        self.txrm_extractor = TxrmWrapper()
        self.dtype_str = dtype
        self.np_dtype = getattr(np, dtype)
        self.image_output = None
        self.ome_metadata = None

    def _apply_reference(self, images, reference):
        floated_and_referenced = [(image * 100.) / reference for image in images]
        referenced_image = []
        for image in floated_and_referenced:
            if np.isnan(image).any() or np.isinf(image).any():
                logging.warning("Potential dead pixels found. "
                                "NaN was output for at least one pixel in the referenced image.")
            referenced_image.append(np.around(image).astype(self.np_dtype))

        return referenced_image

    def _get_reference(self, ole, txrm_name, custom_reference, ignore_reference):
        if custom_reference is not None:
            logging.debug("%s is being referenced using %s and processed.", txrm_name, custom_reference.name)
            reference_path = str(custom_reference)
            try:
                if isOleFile(reference_path):
                    with OleFileIO(reference_path) as ref_ole:
                        references = self.txrm_extractor.extract_all_images(ref_ole)  # should be float for averaging & dividing
                elif ".tif" in reference_path:
                    with tf.TiffFile(reference_path) as tif:
                        references = [page for page in tif.pages[:]]
                else:
                    logging.error("Unable to open file '%s'. Only tif/tiff or xrm/txrm files are supported for custom references.", reference_path)
                    raise IOError(f"Unable to open file '{reference_path}'. Only tif/tiff or xrm/txrm files are supported for custom references.")
            except:
                logging.error("Error occurred reading custom reference", exc_info=True)
                raise
            if len(references) > 1:
                # if reference file is an image stack take median of the images
                return np.median(np.asarray(references), axis=0)
            return references[0]

        elif ole.exists("ReferenceData/Image") and not ignore_reference:
            logging.debug("%s is being referenced and processed using an internal reference.", txrm_name)
            return self.txrm_extractor.extract_reference_image(ole)

        logging.debug("%s is being processed without a reference.", txrm_name)
        return None

    def convert(self, txrm_file, custom_reference, ignore_reference):
        with OleFileIO(str(txrm_file)) as ole:
            images = self.txrm_extractor.extract_all_images(ole)
            reference = self._get_reference(ole, txrm_file.name, custom_reference, ignore_reference)
            if reference is not None:
                self.image_output = self._apply_reference(images, reference)
            else:
                self.image_output = [image for image in np.around(images).astype(self.np_dtype)]
            if (len(self.image_output) > 1
                    and ole.exists("ImageInfo/MosiacRows")
                    and ole.exists("ImageInfo/MosiacColumns")):
                    # Version 13 style mosaic:
                mosaic_rows = self.txrm_extractor.read_imageinfo_as_int(ole, "MosiacRows")
                mosaic_cols = self.txrm_extractor.read_imageinfo_as_int(ole, "MosiacColumns")
                if mosaic_rows != 0 and mosaic_cols != 0:
                    self.image_output = self._stitch_images(self.image_output, (mosaic_cols, mosaic_rows), 1)
    
            # Get image dimensions
            x, y = self.image_output[0].shape
            num_frames = len(self.image_output)
            self.dimensions = (x, y, num_frames)  # X, Y, number of frames (T in tilt series)
    
            # Create metadata
            self.ome_metadata = self._create_ome_metadata(ole, self.dimensions)

    @staticmethod
    def _stitch_images(img_list, mosaic_xy_shape, fast_axis):
        slow_axis = 1 - fast_axis
        logging.debug("Fast axis: %i", fast_axis)
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

    def get_image_and_metadata(self):
        if self.image_output is None:
            logging.warning("Image has not been converted, returning (None, None)")
        return self.image_output, self.ome_metadata

    @staticmethod
    def manual_save(tiff_file, image, metadata=None):
        tiff_path = Path(tiff_file)

        if metadata is not None:
            metadata.image().set_Name(tiff_path.name)
            metadata = metadata.to_xml().encode()

        tiff_dir = tiff_path.resolve().parent
        tiff_dir.mkdir(parents=True, exist_ok=True)
        num_frames = len(image)
        logging.info("Saving image as %s with %i frames", tiff_path.name, num_frames)

        with tf.TiffWriter(str(tiff_path)) as tif:
            tif.save(image[0], photometric='minisblack', description=metadata, metadata={'axes':'XYZCT'})
            for i in range(1, num_frames):
                tif.save(image[i], photometric='minisblack', metadata={'axes':'XYZCT'})

    def save(self, tiff_file):
        if (self.image_output is not None) and (self.ome_metadata is not None):
            self.manual_save(tiff_file, self.image_output, self.ome_metadata)
        else:
            logging.error("Nothing to save! Please convert the image first")
            raise IOError("Nothing to save! Please convert the image first")

    def _create_ome_metadata(self, ole, dimensions):

        # Get metadata variables from ole file:
        exposures = self.txrm_extractor.extract_multiple_exposure_times(ole)

        pixel_size = self.txrm_extractor.extract_pixel_size(ole) * 1.e3  # micron to nm

        physical_img_sizes = []
        physical_img_sizes.append(pixel_size * dimensions[0])
        physical_img_sizes.append(pixel_size * dimensions[1])

        x_positions = [coord * 1.e3 for coord in self.txrm_extractor.extract_x_coords(ole)]  # micron to nm
        y_positions = [coord * 1.e3 for coord in self.txrm_extractor.extract_y_coords(ole)]  # micron to nm

        date_time = datetime.datetime.now().isoformat()  # formatted as: "yyyy-mm-ddThh:mm:ss"
        ox = OMEXML()

        image = ox.image()
        image.set_ID("0")
        image.set_AcquisitionDate(date_time)

        pixels = image.Pixels
        pixels.set_DimensionOrder("XYTZC")
        pixels.set_ID("0")
        pixels.set_PixelType(self.dtype_str)
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

            tiffdata = pixels.tiffdata(count)
            tiffdata.set_FirstC(0)
            tiffdata.set_FirstZ(count)
            tiffdata.set_FirstT(0)
            tiffdata.set_IFD(count)
            tiffdata.set_plane_count(1)

        return ox
