version_info = (1, 2, 3)
__version__ = '.'.join(str(c) for c in version_info)
__author__ = "Thomas Fish"


def convert_and_save(input_path, custom_reference=None, output=None, ignore_reference=False, annotate=False, logging_level="info"):
    """
    Converts xrm or txrm file then saves as it as .ome.tiff file.

    Args:
        input_path (str): Path, can be a file or directory (a directory will batch convert all valid contents).
        custom_reference (str, optional): File path, not valid for batch processing. Defaults to None.
        output (str, optional): Output path, for batch processing output must be a directory otherwise outputs will be saved in the same directory as each input file found. Defaults to None.
        ignore_reference (bool, optional): Ignore any internal reference. Defaults to False.
        annotate (bool, optional): Save additional annotated image (if annotations are available). Defaults to False.
        logging_level (str, optional): Defaults to "info".
    """
    from .run import run
    run(input_path, custom_reference, output, annotate, ignore_reference, logging_level)


def convert(input_file, custom_reference=None, ignore_reference=False):
    """
    Args:
        input_file (str): path to txrm/xrm file to convert
        custom_reference (str, optional): Path to txrm/xrm file to use as reference image. Defaults to None.
        ignore_reference (bool, optional): Ignore any internal reference. Defaults to False.

    Returns:
        list of numpy.ndarrays: list of frames in the image
        omexml-dls object: OME metadata
    """
    from .run import _convert_file
    converter = _convert_file(input_file, custom_reference, ignore_reference, annotate=False)
    return converter.get_image_and_metadata()

def convert_with_annotations(input_file, custom_reference=None, ignore_reference=False):
    """
    Args:
        input_file (str): path to txrm/xrm file to convert
        custom_reference (str, optional): Path to txrm/xrm file to use as reference image. Defaults to None.
        ignore_reference (bool, optional): Ignore any internal reference. Defaults to False.

    Returns:
        numpy.ndarray: image with axis order ZYX
        omexml-dls object: OME metadata
        numpy array or None: Annotated image (if annotations were found) with axis order ZYXC
    """
    from .run import _convert_file
    converter = _convert_file(input_file, custom_reference, ignore_reference, annotate=True)
    output = list(converter.get_image_and_metadata())
    output.append(converter.get_annotated_images())
    return output

def save(output_path, image_list, metadata=None, annotated_image=None):
    """
    Saves image_list as to output formatted as a TIFF. This will be an OME-TIFF if the OME metadata is supplied.
    An annotated image will also be saved as a tiff if the PIL image is suppied.

    Args:
        output_path (str or pathlib.Path): File path to save file (annotated file will have the suffix "_Annotated" applied before the extension)
        image_list (list of numpy.ndarrays): Image to save
        metadata (omexml-dls object, optional): Metadata (will not be applied to annotated image). Defaults to None.
        annotated_image (list of RGB numpy arrays, optional): Annotated images to save. Defaults to None.
    """
    from pathlib import Path
    from .txrm_to_image import manual_save, save_colour

    output_path = Path(output_path)
    manual_save(output_path, image_list, metadata)
    if annotated_image is not None:
        from .txrm_to_image import _convert_output_path_to_annotated_path
        ann_path = _convert_output_path_to_annotated_path(output_path)
        save_colour(annotated_image, ann_path)
