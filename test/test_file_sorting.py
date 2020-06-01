import unittest

import re
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile

from src.txrm2tiff.file_sorting import file_can_be_opened, ole_file_works

class TestFileSorting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fake_temp = NamedTemporaryFile(suffix=".txrm", delete=False)
        cls.fake_file = Path(cls.fake_temp.name)
        cls.real_file = Path("/dls/science/groups/das/ExampleData/B24_test_data/data/2019/cm98765-1/raw/XMv10/test_tomo2_e3C_full.txrm")

    @classmethod
    def tearDownClass(cls):
        cls.fake_temp.file.close()
        cls.fake_file.unlink()

    def test_ole_file_returns_true_for_real_file(self):
        if self.real_file.exists():
            self.assertTrue(ole_file_works(self.real_file))
        else:
            print("Cannot run test without access to dls directories")

    def test_real_file_can_be_opened(self):
        if self.real_file.exists():
            self.assertTrue(file_can_be_opened(self.real_file))
        else:
            print("Cannot run test without access to dls directories")

    def test_ole_file_returns_false_for_fake_file(self):
        self.assertFalse(ole_file_works(self.fake_file))

    def test_fake_file_can_be_opened(self):
        self.assertTrue(file_can_be_opened(self.fake_file))

    def test_nonexistent_file_can_be_opened(self):
        self.assertFalse(file_can_be_opened(Path("/fake/path/oh/no/{}".format(datetime.now().strftime('%Y%m%d_%H%M')))))


if __name__ == "__main__":
    unittest.main()