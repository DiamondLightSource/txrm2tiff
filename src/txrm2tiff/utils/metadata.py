import logging
import numpy as np
from scipy import constants
from ome_types import model
from ome_types.model.simple_types import UnitsLength, UnitsTime, UnitsFrequency, Binning
from ome_types.model.channel import AcquisitionMode, IlluminationType
from ..txrm import abstract
from ..xradia_properties.enums import XrmDataTypes as XDT
from ..info import __version__


dtype_dict = {
    "uint16": "uint16",
    "float32": "float",
    "float64": "double"
    }


def create_ome_metadata(txrm: abstract.AbstractTxrm, filename: str = None) -> model.OME:
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
    tiffdata_list = []
    plane_list = []
    for count in range(shape[0]):
        tiffdata_list.append(model.TiffData(
            first_z=count,
            ifd=count,
            plane_count=1
        ))
        
        plane_list.append(model.Plane(
            the_c=0,
            the_t=0,
            the_z=count,
            delta_t=(txrm.datetimes[count] - txrm.datetimes[0]).total_seconds(),
            delta_t_unit=UnitsTime.SECOND,
            exposure_time=exposures[count],
            position_x=x_positions[count],
            positions_x_unit=UnitsLength.NANOMETER,
            position_y=y_positions[count],
            positions_u_unit=UnitsLength.NANOMETER,
            positions_z=z_positions[count],
            positions_z_unit=UnitsLength.REFERENCEFRAME
        ))

    mean_energy = np.mean(txrm.energies)
    kwargs = {}
    if mean_energy:  
        kwargs["mean_energy"] = 1.e9 * mean_energy / (constants.electron_volt * constants.Planck)
        kwargs["wavelength_unit"] = UnitsLength.NANOMETER
    light_source_settings = model.LightSourceSettings(
        id=txrm.light_source.id,
        **kwargs
    )
    
    detector_settings = model.DetectorSettings(
        id=txrm.detector.id,
        binning=Binning("{0}x{0}".format(txrm.read_stream("ImageInfo/CameraBinning")[0])),
        integration=txrm.read_stream("ImageInfo/FramesPerImage", XDT.XRM_UNSIGNED_INT, strict=False)[0],
        read_out_rate=txrm.read_stream("ImageInfo/ReadoutFreq", XDT.XRM_FLOAT, strict=False)[0],
        read_out_rate_unit=UnitsFrequency.HERTZ,
        zoom=txrm.read_stream("ImageInfo/OpticalMagnification", XDT.XRM_FLOAT, strict=False)[0]
    )

    channel = model.Channel(
        id="Channel:0",
        # Energies are 0 for VLM
        acquisition_mode=AcquisitionMode.OTHER if txrm.energies else AcquisitionMode.BRIGHT_FIELD,
        illumination_type=IlluminationType.TRANSMITTED,
        light_source_settings=light_source_settings,
        detector_settings=detector_settings,
    )

    pixels=model.Pixels(
        id="Pixels:0",
        dimension_order="XYCZT",
        size_x=shape[2],
        size_y=shape[1],
        size_c=1,
        size_z=shape[0],
        size_t=1,
        type="uint16",
        physical_size_x=pixel_size,
        physical_size_x_unit=UnitsLength.NANOMETER,
        physical_size_y=pixel_size,
        physical_size_y_unit=UnitsLength.NANOMETER,
        physical_size_z=1,
        physical_size_z_unit=UnitsLength.REFERENCEFRAME,
        tiff_data_blocks=tiffdata_list,
        planes=plane_list,
        channels=[channel]
        )

    image = model.Image(
            id="Image:0",
            acquisition_date=txrm.datetimes[0].isoformat(),
            description='An OME-TIFF file, converted from an XRM type file by txrm2tiff',
            pixels=pixels,
            instrument_ref=model.InstrumentRef(id=txrm.instrument.id),
            objective_settings=model.ObjectiveSettings(id=txrm.objective.id)
            )

    ome = model.OME(
        creator=f"txrm2tiff {__version__}",
        images=[image],
        instruments=[txrm.instrument]
        )

    return ome
