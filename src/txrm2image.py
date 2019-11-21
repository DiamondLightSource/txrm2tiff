import logging
from xml.etree import ElementTree
import uuid

from olefile import OleFileIO
import datetime
import numpy as np
import tifffile as tf

from txrm_wrapper import TxrmWrapper

from omexml.omexml import OMEXML


class ReferenceApplier:

    def apply(self, images, reference):
        floated_and_referenced = [(image * 100.) / reference for image in images]
        referenced_image = []
        for image in floated_and_referenced:
            if np.isnan(image).any() or np.isinf(image).any():
                logging.warn("Potential dead pixels found. "
                             "NaN was output for at least one pixel in the referenced image.")
            referenced_image.append(np.around(image).astype("uint16"))

        return referenced_image


class TxrmToTiff:

    def __init__(self):
        self.txrm_extractor = TxrmWrapper()
        self.image_divider = ReferenceApplier()
        self.datatype = "uint16"

    def apply_reference(self, images, reference):
        return self.image_divider.apply(images, reference)

    def convert(self, txrm_file, tiff_file, ignore_reference):
        ole = OleFileIO(str(txrm_file))
        images = self.txrm_extractor.extract_all_images(ole)
        if ole.exists("ReferenceData/Image") and not ignore_reference:
            logging.debug("{} is being referenced and processed.".format(txrm_file.name))
            reference = self.txrm_extractor.extract_reference_image(ole)
            image_output = self.apply_reference(images, reference)
        else:
            logging.warning("{} is being processed without a reference.".format(txrm_file.name))
            image_output = [image for image in np.around(images).astype(self.datatype)]

        # Get image dimensions
        x, y = image_output[0].shape
        z = len(image_output)
        dimensions = (x, y, z)  # X, Y, Z

        logging.debug("Saving image as {} with {} frames".format(tiff_file.name, z))

        # Create metadata
        ome_metadata = self.create_ome_metadata(str(tiff_file.name), ole, dimensions)

        with tf.TiffWriter(str(tiff_file)) as tif:
            tif.save(image_output[0], photometric='minisblack', description=ome_metadata, metadata={'axes':'XYCZT'})
            for i in range(1, z):
                tif.save(image_output[i], photometric='minisblack', metadata={'axes':'XYCZT'})
            tif.close()
        ole.close()

    def create_ome_metadata(self, image_name, ole, dimensions):

        # Get metadata variables from ole file:
        exposures = self.txrm_extractor.extract_multiple_exposure_times(ole)

        pixel_size = self.txrm_extractor.extract_pixel_size(ole)

        physical_img_sizes = []
        physical_img_sizes.append(pixel_size * dimensions[0])
        physical_img_sizes.append(pixel_size * dimensions[1])

        x_positions = self.txrm_extractor.extract_x_coords(ole)
        y_positions = self.txrm_extractor.extract_y_coords(ole)

        date_time = datetime.datetime.now().isoformat()  # formatted as: "yyyy-mm-ddThh:mm:ss"
        ox = OMEXML()

        image = ox.image()
        image.set_Name(image_name)
        image.set_ID("0")
        image.set_AcquisitionDate(date_time)

        pixels = image.Pixels

        pixels.set_DimensionOrder("XYCZT")
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

        # Run checks to make sure the value lists are long enough
        exp_len_diff = dimensions[2] - len(exposures)
        if exp_len_diff > 0:
            logging.error("Not enough exposure values for each plane ({} vs {}). Adding zeros to the later planes.".format(len(exposures), dimensions(2)))
            for _ in range(exp_len_diff):
                exposures.append(0)

        x_len_diff = dimensions[2] - len(x_positions)
        if x_len_diff > 0:
            logging.error("Not enough x values for each plane ({} vs {}). Adding zeros to the later planes.".format(len(x_positions), dimensions(2)))
            for _ in range(x_len_diff):
                x_positions.append(0)

        y_len_diff = dimensions[2] - len(y_positions)
        if y_len_diff > 0:
            logging.error("Not enough y values for each plane ({} vs {}). Adding zeros to the later planes.".format(len(y_positions), dimensions(2)))
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
