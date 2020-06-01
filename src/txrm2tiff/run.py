from pathlib import Path, PurePath
import logging

from .file_sorting import file_can_be_opened, ole_file_works
from .logger import create_logger
from .txrm_to_image import TxrmToImage


def run(input_path, custom_reference=None, output_path=None, ignore_reference=False, logging_level="info"):
    create_logger(logging_level.lower())

    input_filepath = Path(input_path)
    if input_filepath.exists():
        if input_filepath.is_dir():
            logging.info("Converting files in directory '%s'", input_path)
            _batch_convert_files(input_filepath, output_path, ignore_reference)
        else:
            logging.info("Converting file '%s'", input_path)
            _convert_and_save(input_filepath, output_path, custom_reference, ignore_reference)
    else:
        logging.error("No such file or directory: %s", input_path)
        raise IOError("No such file or directory: {input_file}")


def _batch_convert_files(input_filepath, output, ignore_reference):
    filepath_list = list(input_filepath.rglob("*xrm"))
    logging.info(f"Batch converting files {[filepath.name for filepath in filepath_list]}")
    if output is not None and Path(output).is_dir():
        logging.info(f"Output directory: {output}")
        output_path_base = Path(output)
        # Replace input_filepath with output, maintaining sub directory structure, and find output file suffix
        output_path_list = [_define_output_suffix(output_path_base / filepath.relative_to(input_filepath)) for filepath in filepath_list]
        for output_path in output_path_list:
            # Make any directories that do not exist:
            output_path.parent.mkdir(parents=True, exist_ok=True)    
    else:
        # Defines output now rather than in _convert_and_save so that there's a list to iterate
        output_path_list = [_define_output_suffix(filepath) for filepath in filepath_list]
        
    for filepath, output_path in zip(filepath_list, output_path_list):
        _convert_and_save(filepath, output_path, None, ignore_reference)


def _convert_and_save(input_filepath, output, custom_reference, ignore_reference):
    converter = _convert_file(input_filepath, custom_reference, ignore_reference)
    if converter is not None:
        output_path = _decide_output(input_filepath, output)
        converter.save(output_path)


def _convert_file(input_filepath, custom_reference, ignore_reference):
    if file_can_be_opened(input_filepath) and ole_file_works(input_filepath):
        logging.info(f"Converting {input_filepath.name}")
        converter = TxrmToImage()
        converter.convert(input_filepath, custom_reference, ignore_reference)
        return converter
    logging.error("Invalid input: '%s'", input_filepath)
        

def _decide_output(input_filepath, output):
    # If no output is supplied
    if output is None:
        output_path = input_filepath
    elif isinstance(output, (str, PurePath)):
        # PurePath is the parent class for all pathlib paths
        output_path = Path(output)
        if output_path.suffix == "" or output_path.is_dir():
            output_path.mkdir(parents=True,exist_ok=True)
            output_path = output_path / input_filepath.name
    else:
        logging.error("Invalid output specified: %s. The input path will be used to create the output.", output)
        output_path = input_filepath
    return _define_output_suffix(output_path, suffix=input_filepath.suffix)

      
def _define_output_suffix(filepath, suffix=None):
    if suffix is None:
        suffix = filepath.suffix
    if suffix == ".txrm":
        return filepath.with_suffix(".ome.tiff")
    elif suffix == ".xrm":
        return filepath.with_suffix(".ome.tif")
    else:
        logging.error("Invalid file extension: %s", suffix)
        raise NameError(f"Invalid file extension: {suffix}")
