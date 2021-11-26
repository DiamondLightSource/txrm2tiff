from __future__ import annotations
from os import PathLike
from typing import Optional, Union, TYPE_CHECKING
from .info import __version__, __author__, __email__

if TYPE_CHECKING:
    from .txrm.abstract import AbstractTxrm




def convert_and_save(
    input_path: PathLike,
    custom_reference: Optional[PathLike] = None,
    output_path: Optional[PathLike] = None,
    annotate: bool = False,
    data_type: Optional[Union[str, int]] = None,
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
        annotate=annotate,
        custom_reference=custom_reference,
        output_path=output_path,
        data_type=data_type,
        ignore_reference=ignore_reference,
        logging_level=logging_level,
    )

from .txrm.main import open_txrm

# def open_txrm(
#     filepath: PathLike,
#     load_images: bool = False,
#     load_reference: bool = False,
#     strict: bool = False,
# ) -> Optional[AbstractTxrm]:
#     from .txrm.main import open_txrm

#     return open_txrm(filepath, load_images, load_reference, strict)
