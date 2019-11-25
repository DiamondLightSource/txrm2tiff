from nose.tools import assert_false, assert_true, assert_equals, with_setup, raises
import re
from pathlib2 import Path
import shutil
from time import sleep, time
import uuid

from file_sorting import *


def setup_files():
    global base
    base = Path(str(uuid.uuid1()))


def teardown_files():
    shutil.rmtree(str(base))


def test_all_frames_written():
    real_file = Path("/dls/science/groups/das/ExampleData/B24_test_data/data"
                     "/2019/cm98765-1/raw/test_tomo2_e3C_full.txrm")
    assert_true(all_frames_written(real_file))


def test_file_can_be_opened():
    real_file = Path("/dls/science/groups/das/ExampleData/B24_test_data/data"
                     "/2019/cm98765-1/raw/test_tomo2_e3C_full.txrm")
    assert_true(file_can_be_opened(real_file))

