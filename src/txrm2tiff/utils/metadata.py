from __future__ import annotations
import numpy as np
from decimal import Decimal
from functools import lru_cache
from ome_types.model.simple_types import PixelType, UnitsLength
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from ome_types import OME

__DTYPE_DICT = {
    "int8": PixelType.INT8,
    "int16": PixelType.INT16,
    "int32": PixelType.INT32,
    "uint8": PixelType.UINT8,
    "uint16": PixelType.UINT16,
    "uint32": PixelType.UINT32,
    "float32": PixelType.FLOAT,
    "float64": PixelType.DOUBLE,
}

# Most advisable dtypes for this data
CLI_DTYPES = ("uint16", "float32", "float64")

__TIFF_TO_OME_UNITS_MAPPING: dict[int, UnitsLength] = {
    1: UnitsLength.REFERENCEFRAME,
    2: UnitsLength.INCH,
    3: UnitsLength.CENTIMETER,
    4: UnitsLength.MILLIMETER,
    5: UnitsLength.MICROMETER,
}

__EXPONENT_UNITS: dict[UnitsLength, int] = {
    UnitsLength.YOTTAMETER: 24,
    UnitsLength.ZETTAMETER: 21,
    UnitsLength.EXAMETER: 18,
    UnitsLength.PETAMETER: 15,
    UnitsLength.TERAMETER: 12,
    UnitsLength.GIGAMETER: 9,
    UnitsLength.MEGAMETER: 6,
    UnitsLength.KILOMETER: 3,
    UnitsLength.HECTOMETER: 2,
    UnitsLength.DECAMETER: 1,
    UnitsLength.METER: 0,
    UnitsLength.DECIMETER: -1,
    UnitsLength.CENTIMETER: -2,
    UnitsLength.MILLIMETER: -3,
    UnitsLength.MICROMETER: -6,
    UnitsLength.NANOMETER: -9,
    UnitsLength.PICOMETER: -12,
    UnitsLength.FEMTOMETER: -15,
    UnitsLength.ATTOMETER: -18,
    UnitsLength.ZEPTOMETER: -21,
    UnitsLength.YOCTOMETER: -24,
    UnitsLength.ANGSTROM: -10,
}

__metres_in_an_inch = Decimal("0.0254")
__MULTIPLIER_UNITS: dict[UnitsLength, Decimal] = {
    UnitsLength.THOU: __metres_in_an_inch / 1000,
    UnitsLength.LINE: __metres_in_an_inch / 12,
    UnitsLength.INCH: __metres_in_an_inch,
    UnitsLength.FOOT: __metres_in_an_inch * 12,
    UnitsLength.YARD: __metres_in_an_inch * 36,
    UnitsLength.MILE: __metres_in_an_inch * 63360,
    UnitsLength.POINT: __metres_in_an_inch / 72,
    # Ignoring:
    # UnitsLength.ASTRONOMICALUNIT
    # UnitsLength.LIGHTYEAR
    # UnitsLength.PARSEC
}


def get_ome_pixel_type(dtype: DTypeLike) -> PixelType:
    try:
        return __DTYPE_DICT[np.dtype(dtype).name]
    except Exception:
        raise TypeError(
            f"{dtype} is unsupported data type. Supported dtypes are {__DTYPE_DICT.keys()}.",
        )


@lru_cache(maxsize=8)  # Unlikely to be many combinations
def _get_multiplier(current_unit: UnitsLength, new_unit: UnitsLength) -> Decimal:
    current_exponent = __EXPONENT_UNITS.get(current_unit, None)
    new_exponent = __EXPONENT_UNITS.get(new_unit, None)
    if current_exponent is not None and new_exponent is not None:
        return Decimal(10) ** (current_exponent - new_exponent)

    # Handle if one or both is a multiplier unit
    if current_exponent is None:
        current_multiplier = __MULTIPLIER_UNITS.get(current_unit, None)
        if current_multiplier is None:
            raise ValueError(f"Invalid unit {current_unit} for conversion")
        multiplier = current_multiplier
    else:
        multiplier = Decimal(10) ** current_exponent
    if new_exponent is None:
        new_multiplier = __MULTIPLIER_UNITS.get(new_unit, None)
        if new_multiplier is None:
            raise ValueError(f"Invalid unit {new_unit} for conversion")
        multiplier /= new_multiplier
    else:
        multiplier *= Decimal(10) ** -new_exponent
    return multiplier


def convert_to_unit(
    value: float, current_unit: UnitsLength, new_unit: UnitsLength
) -> float:
    multipler = _get_multiplier(current_unit, new_unit)
    return float(multipler * Decimal(value))


def handle_tiff_resolution(metadata: OME, resolution_unit: int) -> list[float, float]:
    pixels = metadata.images[0].pixels
    x_size_info = (pixels.physical_size_x, pixels.physical_size_x_unit)
    y_size_info = (pixels.physical_size_y, pixels.physical_size_y_unit)
    if None in x_size_info or None in y_size_info:
        raise ValueError("Failed to interpret physical size from OME metadata")
    try:
        new_unit = __TIFF_TO_OME_UNITS_MAPPING[resolution_unit]
    except KeyError:
        raise ValueError(f"Unit {resolution_unit} is not a supported TIFF unit")
    return [
        1.0 / convert_to_unit(size_info[0], size_info[1], new_unit)
        for size_info in (x_size_info, y_size_info)
    ]
