import unittest
import numpy as np

from txrm2tiff.utils.functions import conditional_replace, convert_to_int


class TestUtilsFunctions(unittest.TestCase):
    def test_conditional_replace(self):
        array = np.repeat(np.arange(0, 9000, 0.1).reshape(300, 300), 10, 0)
        threshold = 100

        conditional_replace(array, np.nan, lambda x: x < threshold)

        self.assertEqual(
            np.nanmin(array),
            threshold,
            msg="Array values below threshold {} have not been replaced with nan".format(
                threshold
            ),
        )

    def test_convert_to_int(self):
        for i in range(0, 320000):
            self.assertEqual(convert_to_int(float(i)), i)

    def test_convert_to_int_fails(self):
        with self.assertRaises(ValueError):
            convert_to_int(3.2)
