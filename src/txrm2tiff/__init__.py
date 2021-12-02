from os import PathLike
from typing import Optional, Union
from .info import __version__, __author__, __email__

from .txrm.main import open_txrm


def convert_and_save(
    input_path: PathLike,
    custom_reference: Optional[PathLike] = None,
    output_path: Optional[PathLike] = None,
    annotate: bool = False,
    flip: bool = False,
    data_type: Optional[Union[str, int]] = None,
    ignore_shifts: bool = False,
    ignore_reference: bool = False,
    logging_level: Union[str, int] = "info",
):
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
    from .run import run

    run(
        input_path=input_path,
        output_path=output_path,
        custom_reference=custom_reference,
        annotate=annotate,
        flip=flip,
        data_type=data_type,
        ignore_shifts=ignore_shifts,
        ignore_reference=ignore_reference,
        logging_level=logging_level,
    )
