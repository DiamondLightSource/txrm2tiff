import unittest
from parameterized import parameterized

import numpy as np
from ome_types.model.simple_types import UnitsLength
from txrm2tiff.utils import metadata


class TestMetadataFunctions(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "both expontents",
                UnitsLength.NANOMETER,
                UnitsLength.CENTIMETER,
                50.0,
                5.0e-6,
            ),
            ("current multipler", UnitsLength.INCH, UnitsLength.CENTIMETER, 5.0, 12.7),
            (
                "new multipler",
                UnitsLength.KILOMETER,
                UnitsLength.MILE,
                21.5,
                13.3594806,
            ),
            ("both multiplier", UnitsLength.MILE, UnitsLength.YARD, 2.5, 4400),
        ]
    )
    def test_convert_to_unit(
        self, _name, current_unit, new_unit, value, expected_value
    ):
        self.assertAlmostEqual(
            metadata.convert_to_unit(value, current_unit, new_unit), expected_value
        )
