from txrm_wrapper import TxrmWrapper
from mock import MagicMock, patch
import struct
import numpy as np
from nose.tools import assert_equal, assert_true, assert_almost_equal
from scipy.constants import h, c, e


def test_extracting_single_image():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream
    ole.exists.return_value = True
    stream.getvalue.return_value = struct.pack('<6H', *range(6))

    data = TxrmWrapper().extract_single_image(ole, 1, 2, 3)

    ole.openstream.assert_called_with('ImageData1/Image1')

    assert (data == np.arange(6).reshape(2, 3)).all(), "output is not as expected"


def test_read_stream_failure():
    ole = MagicMock()
    ole.exists.return_value = False
    data = TxrmWrapper().read_stream(ole, "key", 'i')

    assert_true(data is None)


def test_extracts_dimensions():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream
    stream.getvalue.side_effect = [pack_int(i) for i in [6, 7]]

    data = TxrmWrapper().extract_image_dims(ole)

    assert_equal(data, (6, 7))


def test_extracts_number_of_images():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream
    stream.getvalue.side_effect = [pack_int(i) for i in [9]]

    data = TxrmWrapper().extract_number_of_images(ole)

    assert_equal(data, 9)


def test_extracts_multiple_images():
    ole = MagicMock()
    stream = MagicMock()
    ole.exists.side_effect = ([True] * 9) + [False]
    ole.openstream.return_value = stream
    dimensions = [pack_int(i) for i in [2, 3]]
    images = [struct.pack('<6H', *range(6))] * 5
    images_taken = [pack_int(5)]
    stream.getvalue.side_effect = dimensions + images_taken + images

    data = TxrmWrapper().extract_all_images(ole)

    assert len(data) == 5


def test_extracts_tilt_angles():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream
    packed_tilts = struct.pack('<4f', 1, 2, 3, 4)
    stream.getvalue.side_effect = [packed_tilts]

    data = TxrmWrapper().extract_tilt_angles(ole)
    ole.openstream.assert_called_with('ImageInfo/Angles')
    assert_true((data == np.array([1, 2, 3, 4])).all())


def test_extacts_exposure_time_tomo():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream
    # Testing uneven tilt (more +ve than -ve values):
    packed_exposures = struct.pack('<9f', 1, 2, 3, 4, 5, 6, 7, 8, 9)
    packed_angles = struct.pack('<9f', -3, -2, -1, 0, 1, 2, 3, 4, 5)
    stream.getvalue.side_effect = [packed_exposures, packed_angles]
    ole.exists.return_value = True
    data = TxrmWrapper().extract_exposure_time(ole)

    assert_equal(data, 4.)


def test_extacts_exposure_time_single_image():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream
    packed_exposure = struct.pack('<1f', 5)
    stream.getvalue.side_effect = [packed_exposure]
    ole.exists.side_effect = [False, True, True]
    data = TxrmWrapper().extract_exposure_time(ole)

    assert_equal(data, 5)


def test_extacts_pixel_size():
    ole = create_ole_that_returns_float()
    data = TxrmWrapper().extract_pixel_size(ole)

    assert_equal(data, 100.5)


def test_extracts_xray_magnification():
    ole = create_ole_that_returns_float()
    data = TxrmWrapper().extract_xray_magnification(ole)

    assert_equal(data, 100.5)


def test_extracts_energy():
    ole = create_ole_that_returns_float()
    data = TxrmWrapper().extract_energy(ole)

    assert_equal(data, 100.5)


def test_extracts_wavelength():
    ole = create_ole_that_returns_float()
    data = TxrmWrapper().extract_wavelength(ole)
    assert_equal(float("%.8e" % data), 1.23367361e-8)


@patch('txrm_wrapper.TxrmWrapper.read_stream')
def test_create_mosaic_of_reference_image(read_stream):
    reference_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    mosaic_reference = TxrmWrapper().create_reference_mosaic(MagicMock(), reference_data, 6, 3, 2, 1)
    expected_reference = np.array([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9],
                                   [1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]
                                   ])
    assert(mosaic_reference.shape == expected_reference.shape), "Arrays must be the same shape"
    assert_true((mosaic_reference == expected_reference).all())


def test_rescale_ref_exposure():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream
    # Testing uneven tilt (more +ve than -ve values):
    ref_exposure = struct.pack('<1f', 2)
    packed_exposures = struct.pack('<9f', 1, 2, 3, 4, 5, 6, 7, 8, 9)
    packed_angles = struct.pack('<9f', -3, -2, -1, 0, 1, 2, 3, 4, 5)
    stream.getvalue.side_effect = [ref_exposure, packed_exposures, packed_angles]
    ole.exists.return_value = True

    assert_true((
        TxrmWrapper().rescale_ref_exposure(ole, np.array([2., 4., 6.])) ==
        np.array([1., 2., 3.])).all)


def create_ole_that_returns_integer():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream

    packed_pixel_size = struct.pack('<I', 100)
    stream.getvalue.side_effect = [packed_pixel_size]
    return ole


def create_ole_that_returns_float():
    ole = MagicMock()
    stream = MagicMock()
    ole.openstream.return_value = stream

    packed_pixel_size = struct.pack('<f', 100.5)
    stream.getvalue.side_effect = [packed_pixel_size]
    return ole


def pack_int(number):
    return struct.pack('<I', np.int(number))
