from os import PathLike
from typing import Optional, Union
from .info import __version__, __author__, __email__

from .txrm.main import open_txrm
from .main import convert_and_save
