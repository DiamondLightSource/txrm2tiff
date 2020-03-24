from nose.tools import with_setup, assert_true
import numpy as np
import os
import logging
from pathlib import Path

from txrm2image import TxrmToTiff
from txrm_wrapper import TxrmWrapper

real_directory = ("/dls/science/groups/das/ExampleData/B24_test_data/data"
                  "/2019/cm98765-1/raw")

real_file_list = [
    real_directory + '/XMv10/12_Tomo_F4D_Area1_noref.txrm',
    real_directory + '/XMv10/VLM_mosaic.xrm',
    real_directory + '/XMv10/test_tomo2_e3C_full.txrm',
    real_directory + '/XMv10/Xray_mosaic_F5A.xrm',
    real_directory + '/XMv13/VLM_mosaic_v13_interrupt.xrm',
    real_directory + '/XMv13/Xray_mosaic_v13.xrm',
    real_directory + '/XMv13/tomo_v13_interrupt.txrm',
    real_directory + '/XMv13/Xray-single-v13-reference-image.xrm'
]


def setup():
    pass


def output_teardown():
    output_dir = real_directory.replace('raw', 'processed')
    output_files = os.listdir(output_dir)
    for file in output_files:
        os.remove(os.path.join(output_dir, file))


def test_divides_images_by_reference():
    images = [np.array([[0, 2, 4], [6, 8, 10]])] * 5
    reference = [np.arange(6).reshape(2, 3)] * 5
    result = TxrmToTiff().apply_reference(images, reference)
    assert len(result) == 5, "The result is the wrong length"
    assert (result[3] == np.array([[0, 200, 200], [200, 200, 200]])).all(), "The result does not match the expected result"


@with_setup(setup, output_teardown)
def test_converts_to_tiff():
    for test_file in real_file_list:
        logging.debug("Running with file {}".format(test_file))
        output_file = Path(test_file).with_suffix('.ome.tiff')
        converter = TxrmToTiff()
        converter.convert(Path(test_file), Path(output_file), None, False)
        assert_true(output_file.exists())


@with_setup(setup, output_teardown)
def test_can_apply_custom_reference():
    test_file = real_file_list[2]
    reference_file = real_file_list[0]
    logging.debug("Running with file {}".format(test_file))
    output_file = Path(test_file).with_suffix('.ome.tiff')
    converter = TxrmToTiff()
    converter.convert(Path(test_file), Path(output_file), Path(reference_file), False)
    assert_true(output_file.exists())

