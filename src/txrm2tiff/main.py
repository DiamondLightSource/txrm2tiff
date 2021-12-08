from os import PathLike
from pathlib import Path
import logging
from typing import Optional, Union
from numpy.typing import DTypeLike

from .txrm import open_txrm
from .txrm.abstract import AbstractTxrm
from .utils.logging import create_logger


def convert_and_save(
    input_path: Union[str, PathLike],
    output_path: Optional[Union[str, PathLike]] = None,
    custom_reference: Optional[Union[str, PathLike]] = None,
    annotate: bool = False,
    flip: bool = False,
    data_type: Optional[str] = None,
    ignore_shifts: bool = False,
    ignore_reference: bool = False,
    logging_level: Union[str, int] = "info",
) -> None:
    """
    Converts xrm or txrm file then saves as it as .ome.tiff file.

    Args:
        input_path (str): Path, can be a file or directory (a directory will batch convert all valid contents).
        custom_reference (str, optional): File path, not valid for batch processing. Defaults to None.
        output_path (str, optional): Output path, for batch processing output must be a directory otherwise outputs will be saved in the same directory as each input file found. Defaults to None.
        annotate (bool, optional): Save additional annotated image (if annotations are available). Defaults to False.
        ignore_reference (bool, optional): Ignore any internal reference. Defaults to False.
        logging_level (str, optional): Defaults to "info".
    """
    create_logger(str(logging_level).lower())
    logging.debug(
        "Running with arguments: "
        "input_path=%s, custom_reference=%s, output_path=%s, annotate=%s, flip=%s, data_type=%s, ignore_shifts=%s, ignore_reference=%s, logging_level=%s",
        input_path,
        custom_reference,
        output_path,
        annotate,
        flip,
        data_type,
        ignore_shifts,
        ignore_reference,
        logging_level,
    )
    input_path = Path(input_path)
    if input_path.exists():
        if output_path is not None:
            output_path = Path(output_path)
        if custom_reference is not None:
            logging.warning("Custom references are invalid for batch conversions")
        if input_path.is_dir():
            logging.info("Converting files in directory '%s'", input_path)
            _batch_convert_files(
                input_path,
                output_path,
                annotate,
                flip,
                data_type,
                not ignore_shifts,
                ignore_reference,
            )
        else:
            if custom_reference is not None:
                custom_reference = Path(custom_reference)
                if not custom_reference.is_file():
                    logging.warning(
                        "The specified custom reference file '%s' does not exist.",
                        custom_reference,
                    )
                    custom_reference = None
            logging.info("Converting file '%s'", input_path)
            _convert_and_save(
                input_path,
                output_path,
                custom_reference,
                annotate,
                flip,
                data_type,
                not ignore_shifts,
                ignore_reference,
            )
    else:
        logging.error("No such file or directory: %s", input_path)
        raise IOError(f"No such file or directory: {input_path}")


def _batch_convert_files(
    input_directory: Path,
    output: Optional[Path] = None,
    annotate: bool = True,
    flip: bool = False,
    data_type: Optional[DTypeLike] = None,
    shifts: bool = True,
    ignore_reference: bool = False,
) -> None:
    filepath_list = _find_files(input_directory)
    logging.info(
        f"Batch converting files: {', '.join([filepath.name for filepath in filepath_list])}"
    )
    if output is not None:
        logging.info(f"Output directory set to: {output}")
        output_path_base = Path(output)
        # Replace input_filepath with output, maintaining sub directory structure, and find output file suffix
        output_path_list = [
            _set_output_suffix(output_path_base / filepath.relative_to(input_directory))
            for filepath in filepath_list
        ]
        for output_path in output_path_list:
            # Make any directories that do not exist:
            output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Defines output now rather than in _convert_and_save so that there's a list to iterate
        output_path_list = [_set_output_suffix(filepath) for filepath in filepath_list]

    for filepath, output_path in zip(filepath_list, output_path_list):
        _convert_and_save(
            filepath,
            output_path,
            None,
            annotate,
            flip,
            data_type,
            shifts,
            ignore_reference,
        )


def _find_files(directory: Path):
    return list(directory.rglob("*xrm"))


def _convert_and_save(
    input_path: Path,
    output_path: Optional[Path] = None,
    custom_reference: Optional[Union[str, PathLike]] = None,
    annotate: bool = True,
    flip: bool = False,
    data_type: Optional[DTypeLike] = None,
    shifts: bool = True,
    ignore_reference: bool = False,
) -> None:
    with open_txrm(input_path) as txrm:
        _convert_file(txrm, custom_reference, ignore_reference, annotate)
        output_path = _decide_output_path(txrm.path, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Make output directory
        txrm.save_images(output_path, data_type, flip=flip, shifts=shifts, mkdir=True)


def _convert_file(
    txrm: AbstractTxrm,
    custom_reference: Optional[Union[str, PathLike]] = None,
    ignore_reference: bool = False,
    annotate: bool = True,
) -> None:
    logging.info(f"Converting {txrm.name}")
    # custom_reference overrules ignore_reference
    if custom_reference is not None or not ignore_reference:
        txrm.apply_reference(
            custom_reference
        )  # Called with None applies internal reference
    if annotate:
        txrm.annotate()


def _decide_output_path(input_path: Path, output_path: Optional[Path]) -> Path:
    # If no output is supplied
    if output_path is None:
        output_path = input_path
    else:
        if output_path.suffix == "" or output_path.is_dir():
            output_path = output_path / input_path.name
        else:
            # If a valid filepath was given, just return that (avoids double .ome)
            return output_path
    return _set_output_suffix(output_path, curent_suffix=input_path.suffix)


def _set_output_suffix(filepath: Path, curent_suffix: Optional[str] = None) -> Path:
    if curent_suffix is None:
        curent_suffix = filepath.suffix
    if curent_suffix == ".txrm":
        return filepath.with_suffix(".ome.tiff")
    elif curent_suffix == ".xrm":
        return filepath.with_suffix(".ome.tif")
    else:
        logging.error("Invalid file extension: %s", curent_suffix)
        raise NameError(f"Invalid file extension: {curent_suffix}")
