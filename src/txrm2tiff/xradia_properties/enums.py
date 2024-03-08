import enum
import numpy as np
from types import DynamicClassAttribute


class IntValueEnum(int, enum.Enum):
    """
    An enumerator base class that allows the setting of both a value and label.
    IntEnum(value) returns the as expected

    Based on https://stackoverflow.com/a/68400507
    """

    def __new__(cls, value, number: int):
        obj = int.__new__(cls, number)
        obj._value_ = value
        obj.number = number
        return obj

    def __str__(self):
        return f"{self.name} ({self.value.__name__})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, {self.value.__name__}, {self.number})"

    def __format__(self, format_spec):
        return str(self)

    @classmethod
    def from_number(cls, number: int, strict=False):
        for obj in cls:
            if obj.number == number:
                return obj
        if strict:
            raise ValueError(f"{cls.__name__} has no value matching {number}")
        return None


class XrmDataTypes(IntValueEnum):
    """Integer Enumerator for Xradia's XrmDataTypes"""

    # XRM_BIT = None, 1
    XRM_CHAR = np.int8, 2
    XRM_UNSIGNED_CHAR = np.uint8, 3
    XRM_SHORT = np.int16, 4
    XRM_UNSIGNED_SHORT = np.uint16, 5
    XRM_INT = np.int32, 6
    XRM_UNSIGNED_INT = np.uint32, 7
    XRM_LONG = np.int64, 8
    XRM_UNSIGNED_LONG = np.uint64, 9
    XRM_FLOAT = np.float32, 10
    XRM_DOUBLE = np.float64, 11
    XRM_STRING = np.str_, 12
    # XRM_DATATYPE_SIZE = None, 13


class AnnotationTypes(enum.Enum):
    ANN_LINE = 0
    ANN_RECT = 1
    ANN_ELLIPSE = 2
    ANN_CIRCLE = 3
    ANN_TEXT = 4  # Doesn't work in XRMDataExplorer
    ANN_POLYGON = 5
    ANN_GROUP = 6
    ANN_LINE_PLOT = 7
    ANN_RULE = 8
    ANN_ANGLE = 9
    ANN_MOUSEPOINT = 10
    ANN_CONE_ANGLE_LINE = 11
    ANN_HIGHLIGHTER = 12
    ANN_POLYLINE = 13
    ANN_FREE_HAND_SKETCH = 14
    ANN_SIZE = 15


class XrmCameraType(enum.Enum):
    ANDOR = 0
    PICOLO = 1
    PIXIS = 2
    RETIGA = 3
    UEYE = 4
    WEBCAM = 5
    SIMCAM = 6
    DEXELA = 7
    PCO_EDGE = 8
    COUNT_CAM_TYPE = 9


class XrmSourceType(enum.Enum):
    XRM_RIGAKU_SOURCE = 0
    XRM_HAMAMATSU_100 = 1
    XRM_HAMAMATSU_150 = 2
    XRM_KEVEX_133 = 3
    XRM_NO_SOURCE = 4  # Synchrotron
    XRM_HAMAMATSU_90 = 5
    XRM_HAMAMATSU_100_20W = 6
    XRM_KEVEX_90KV_8W = 7
    XRM_SOURCE_DAGE_MARKIII = 8
    XRM_HAMAMATSU_150_30W = 9
    XRM_XRAYSOURCE_SIMULATED_DO_NOT_USE = (
        10  # obsolete, set simulation flag in config instead
    )
    XRM_RIGAKU_SIMULATED_DO_NOT_USE = 11  # obsolete, use emulated XRM_RIGAKU_SOURCE
    XRM_HAMAMATSU_150_V2 = 12
    XRM_HAMAMATSU_150_30W_V2 = 13
    XRM_ENERGETIQ_SOURCE = 14
    XRM_VISUAL_LIGHT_SOURCE = 15
    XRM_SOURCE_DAGE_MARKIV_LP = 16
    XRM_SOURCE_DAGE_MARKIV_HP = 17
    XRM_ARTESIAN = 18
    XRM_RIGAKU_SOURCE_V2 = 19
