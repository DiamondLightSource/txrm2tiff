version_info = (1, 0, 1)
__version__ = '.'.join(str(c) for c in version_info)
__author__ = "Thomas Fish"


def convert_and_save(input_path, custom_reference=None, output=None, ignore_reference=False, logging_level="info"):
    """
    Arguments:
        input_path (path, can be a file or directory (a directory will batch convert all valid contents))
        custom_reference=None (file path, not valid for batch processing)
        output=None (path, for batch processing output must be a directory otherwise outputs will be saved in the same directory as each input file found)
        ignore_reference=False (boolean)
        logging_level="info" (string: debug, info, warning, error, critical)

    Saves as .ome.tiff to output
    """
    from .run import run
    run(input_path, custom_reference, output, ignore_reference, logging_level)


def convert(input_file, custom_reference=None, ignore_reference=False, logging_level="info"):
    """
    Arguments:
        input_file (file path)
        custom_reference=None (file path)
        ignore_reference=False (boolean)
        logging_level="info" (string: debug, info, warning, error, critical)

    Outputs:
        image_list (list of numpy arrays)
        metadata (OME metadata as omexml-dls object)
    """
    from .run import _convert_file
    converter = _convert_file(input_file, custom_reference, ignore_reference, logging_level)
    return converter.get_image_and_metadata()


def save(output_path, image_list, metadata=None):
    """
    Arguments:
        output_path (file path)
        image_list (list of numpy arrays)
        metadata=None (OME metadata as omexml-dls object)

    Saves as .ome.tiff to output
    """
    from .txrm_to_image import TxrmToImage
    TxrmToImage.manual_save(output_path, image_list, metadata)
