from pathlib import Path
from file_sorting import *
from logger import create_logger


def run(input, custom_reference=None, output=None, ignore_reference=False, logging_level="info"):
    create_logger(logging_level.lower())

    input_filepath = Path(input)
    if input_path.is_dir():
        _batch_convert_files(input_filepath, output, ignore_reference)
    else:
        _convert_file(input_filepath, output, custom_reference, ignore_reference)

def _batch_convert_files(input_filepath, output, ignore_reference):
    filepath_list = list(input_path.rglob("*xrm"))
    logging.info(f"Batch converting files {*[filepath.name for filepath in filepath_list]}")
    if output is not None and Path(output).is_dir():
        logging.info(f"Output directory: {output}")
        output_path_base = Path(output)
        # Replace input_filepath with output, maintaining sub directory structure, and find output file suffix
        output_path_list = [_define_output_suffix(output_path_base / filepath.relative_to(input_filepath)) for filepath in filepath_list]
        for output_path in output_path_list:
            # Make any directories that do not exist:
            output_path.parent.mkdir(parents=True, exist_ok=True)    
    else:
        # Defines output now rather than in _convert_file so that there's a list to iterate
        output_path_list = [_define_output_suffix(filepath) for filepath in filepath_list]
        
    for filepath, output_path in zip(filepath_list, output_path_list):
        _convert_file(filepath, output_path, None, ignore_reference)

def _convert_file(input_filepath, output, custom_reference, ignore_reference):
    logging.info(f"Running txrm2tiff on {input_filepath.name}")
    if file_can_be_opened(input_filepath) and ole_file_works(input_filepath):

        # If no output is supplied:
        if output is None:
            output_filepath = _define_output_suffix(input_filepath)
        elif isinstance(output, str):
            output_filepath = Path(output)
        else:
            output_filepath = output
        TxrmToTiff().convert(input_filepath, output_filepath, custom_reference, ignore_reference)
        
def _define_output_suffix(input_filepath):    
    output_filepath = input_filepath
    if input_filepath.suffix == ".txrm":
        output_filepath.with_suffix(".ome.tiff")
    elif input_filepath.suffix == ".xrm":
        output_filepath.with_suffix(".ome.tif")
    return output_filepath