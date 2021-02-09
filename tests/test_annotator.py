import unittest
from unittest.mock import MagicMock, patch
from parameterized import parameterized

import numpy as np
from numpy.testing import assert_array_equal
from pathlib import Path

from txrm2tiff.txrm_to_image import TxrmToImage, save_colour
from txrm2tiff import annotator

test_data_path = Path("/dls/science/groups/das/ExampleData/B24_test_data/")
test_file = test_data_path / "annotation_test" / "Xray_mosaic_F3C.xrm"

visit_path = test_data_path / "data" / "2019" / "cm98765-1"
raw_path = visit_path / "raw"
xm10_path = raw_path / "XMv10"
xm13_path = raw_path / "XMv13"

test_files = [
    (test_file, ),
    (xm13_path / 'Xray_mosaic_v13.xrm', ),
    (xm13_path / 'Xray_mosaic_v13_interrupt.xrm', ),
    (xm13_path / 'Xray_single_v13.xrm', ),
    (xm13_path / 'tomo_v13_full.txrm', ),
    (xm13_path / 'tomo_v13_full_noref.txrm', ),
    (xm13_path / 'tomo_v13_interrupt.txrm', ),
    (xm13_path / 'VLM_mosaic_v13.xrm', ),
    (xm13_path / 'VLM_mosaic_v13_interrupt.xrm', ),
    (xm10_path / '12_Tomo_F4D_Area1_noref.txrm', ),
    (xm10_path / 'VLM_mosaic.xrm', ),
    (xm10_path / 'test_tomo2_e3C_full.txrm', ),
    (xm10_path / 'Xray_mosaic_F5A.xrm', ),]


class TestAnnotator(unittest.TestCase):

    @unittest.skipUnless(visit_path.exists(), "dls paths cannot be accessed")
    @parameterized.expand(test_files)
    def test_with_real_image(self, test_file):
        output_file = test_file.parent / (test_file.stem + "_Annotated.tif")

        converter = TxrmToImage()
        converter.convert(test_file, custom_reference=None, ignore_reference=False, annotate=True)
        ann_images = converter.get_annotated_images()
        self.assertFalse(ann_images is None, msg="Image wasn't created")
        save_colour(output_file, ann_images)
        self.assertTrue(output_file.exists(), msg=f"File {output_file} doesn't exist")
        output_file.unlink()

    @patch('txrm2tiff.annotator.txrm_wrapper.read_stream')
    def test_square(self, mocked_stream_reader):
        fill = 125
        im_size = 5
        im = np.zeros((1, im_size, im_size))

        x0, y0 = 1, 1
        x1, y1 = 4, 4

        ann = annotator.Annotator(im[0].shape[::-1])

        with patch.object(ann, "_get_colour") as patched_colour:
            with patch.object(ann, "_get_thickness") as patched_thickness:
                patched_colour.return_value = (0, fill, 0, 255)
                patched_thickness.return_value = 1
                mocked_stream_reader.side_effect = [[x0], [y0], [x1], [y1]]
                ole = MagicMock()
                stream_stem = ""
                ann._plot_rect(ole, stream_stem)
        ann._saved_annotations = True
        output = ann.apply_annotations(im)

        expected_output = np.zeros((im_size, im_size, 3), dtype=np.uint8)
        y0, y1 = im_size - 1 - y0, im_size - 1 - y1  # Invert y axis
        expected_output[y0, x0:x1 + 1, 1] = fill
        expected_output[y1:y0 + 1, x0, 1] = fill
        expected_output[y1, x0:x1 + 1, 1] = fill
        expected_output[y1:y0 + 1, x1, 1] = fill
        
        assert_array_equal(output[0], expected_output)

    @patch('txrm2tiff.annotator.txrm_wrapper.read_stream')
    def test_line(self, mocked_stream_reader):
        fill = 125
        x0, x1 = 1, 4
        y = 4
        im_size = 5
        im = np.zeros((1, im_size, im_size))

        ann = annotator.Annotator(im[0].shape[::-1])

        with patch.object(ann, "_get_colour") as patched_colour:
            with patch.object(ann, "_get_thickness") as patched_thickness:
                patched_colour.return_value = (0, fill, 0, 255)
                patched_thickness.return_value = 1
                mocked_stream_reader.side_effect = [[x0], [x1], [y], [y]]
                ole = MagicMock()
                stream_stem = ""
                ann._plot_line(ole, stream_stem)
        ann._saved_annotations = True
        output = ann.apply_annotations(im)

        expected_output = np.zeros((im_size, im_size, 3), dtype=np.uint8)
        y = im_size - 1 - y  # Invert y axis
        expected_output[y, x0:x1 + 1, 1] = fill

        assert_array_equal(output[0], expected_output)

    @patch('txrm2tiff.annotator.txrm_wrapper.read_stream')
    def test_ellipse(self, mocked_stream_reader):
        fill = 125
        xs = 0, 18
        ys = 2, 16
        x_mid = int(round(np.mean(xs)))
        y_mid = int(round(np.mean(ys)))
        im_size = 20
        im = np.zeros((1, im_size, im_size))

        ann = annotator.Annotator(im[0].shape[::-1])

        with patch.object(ann, "_get_colour") as patched_colour:
            with patch.object(ann, "_get_thickness") as patched_thickness:
                patched_colour.return_value = (0, fill, 0, 255)
                patched_thickness.return_value = 1
                mocked_stream_reader.side_effect = [[xs[0]], [ys[0]], [xs[1]], [ys[1]]]
                ole = MagicMock()
                stream_stem = ""
                ann._plot_ellipse(ole, stream_stem)
        ann._saved_annotations = True
        output = ann.apply_annotations(im)[0]

        self.assertTrue((output[:, :, 0] == 0).any())  # Red
        self.assertTrue((output[:, :, 2] == 0).any())  # Blue
        for (x, y) in ann._flip_y((x_mid, ys[0]), (xs[0], y_mid), (x_mid, ys[1]), (xs[1], y_mid)):
            self.assertEqual(output[y, x, 1], fill, msg="Failed for y, x of %i, %i" %(y, x))
    
    @patch('txrm2tiff.annotator.txrm_wrapper.read_stream')
    def test_polyline(self, mocked_stream_reader):
        fill = 125
        im_size = 18  # Must be an even number
        xs = tuple(range(im_size))
        ys = [0, im_size - 1] * (im_size // 2)
        total_points = len(xs)
        im = np.zeros((1, im_size, im_size))

        ann = annotator.Annotator(im[0].shape[::-1])

        with patch.object(ann, "_get_colour") as patched_colour:
            with patch.object(ann, "_get_thickness") as patched_thickness:
                patched_colour.return_value = (0, fill, 0, 255)
                patched_thickness.return_value = 1
                mocked_stream_reader.side_effect = [[total_points], xs, ys]
                ole = MagicMock()
                stream_stem = ""
                ann._plot_polyline(ole, stream_stem)
        ann._saved_annotations = True
        output = ann.apply_annotations(im)[0]

        self.assertTrue((output[:, :, 0] == 0).any())  # Red
        self.assertTrue((output[:, :, 2] == 0).any())  # Blue
        for x, y in ann._flip_y(*zip(xs, ys)):
            self.assertEqual(output[y, x, 1], fill)

    @patch('txrm2tiff.annotator.txrm_wrapper.read_stream')
    def test_scale_bar(self, mocked_stream_reader):
        im_size = 5
        im = np.zeros((1, im_size, im_size))

        ann = annotator.Annotator(im[0].shape[::-1])
        mocked_stream_reader.side_effect = [[17.]]
        ann._add_scale_bar(MagicMock())
        ann._saved_annotations = True
        output = ann.apply_annotations(im)[0]

        self.assertTrue((output[:, :, 0] == 0).any())    # Red
        self.assertTrue((output[:, :, 1] == 255).any())  # Green
        self.assertTrue((output[:, :, 2] == 0).any())    # Blue


    def test_flip_y(self):
        ann = annotator.Annotator((20, 20))
        res = ann._flip_y((0, 0), (19, 19))
        self.assertEqual(res, ((0, 19), (19, 0)))
