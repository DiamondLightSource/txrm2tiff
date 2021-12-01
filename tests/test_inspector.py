import unittest
from parameterized import parameterized

from pathlib import Path
from random import randint
from txrm2tiff import open_txrm


from txrm2tiff.inspector import Inspector

visit_path = Path(
    "/dls/science/groups/das/ExampleData/B24_test_data/data/2019/cm98765-1"
)
raw_path = visit_path / "raw"
xm10_path = raw_path / "XMv10"
xm13_path = raw_path / "XMv13"

test_files = [
    (xm13_path / "Xray_mosaic_v13.xrm",),
    (xm13_path / "Xray_mosaic_v13_interrupt.xrm",),
    (xm13_path / "Xray_single_v13.xrm",),
    (xm13_path / "tomo_v13_full.txrm",),
    (xm13_path / "tomo_v13_full_noref.txrm",),
    (xm13_path / "tomo_v13_interrupt.txrm",),
    (xm13_path / "VLM_mosaic_v13.xrm",),
    (xm13_path / "VLM_mosaic_v13_interrupt.xrm",),
    (xm10_path / "12_Tomo_F4D_Area1_noref.txrm",),
    (xm10_path / "VLM_mosaic.xrm",),
    (xm10_path / "test_tomo2_e3C_full.txrm",),
    (xm10_path / "Xray_mosaic_F5A.xrm",),
]


@unittest.skipUnless(visit_path.exists(), "dls paths cannot be accessed")
class TestInspectorFunctions(unittest.TestCase):
    @parameterized.expand(test_files)
    def test_basic_inspect(self, test_file):
        with open_txrm(test_file) as txrm:
            inspector = Inspector(txrm)
            inspector.inspect(False)
            text = inspector.get_text()
        self.assertNotIn("This is not a valid txrm/xrm file.", text)

    @parameterized.expand(test_files)
    def test_extra_inspect(self, test_file):
        with open_txrm(test_file) as txrm:
            inspector = Inspector(txrm)
            inspector.inspect(True)
        text = inspector.get_text()
        self.assertNotIn("This is not a valid txrm/xrm file.", text)

    @parameterized.expand(test_files)
    def test_list_streams(self, test_file):
        filename = Path(test_file).name
        with open_txrm(test_file) as txrm:
            inspector = Inspector(txrm)
            inspector.list_streams()
            # get all bits of text that aren't empty or the filename
        text_list = [i for i in inspector.get_text().split("\n") if i and i != filename]
        # There should be well over 10 lines
        self.assertTrue(len(text_list) > 10)

    @parameterized.expand(test_files)
    def test_read_streams(self, test_file):
        with open_txrm(test_file) as txrm:
            inspector = Inspector(txrm)
            stream_list = txrm.list_streams()
            indexes = [randint(0, len(stream_list) - 1) for _ in range(0, 10)]
            # Test a random assortment of keys
            for stream_key in [stream_list[i] for i in indexes]:
                inspector.inspect_streams(stream_key)
        text = inspector.get_text()
        self.assertNotIn(
            "does not exist", text, msg="stream not read despite being in list"
        )
