import unittest
from unittest.mock import MagicMock, call, patch
from parameterized import parameterized

import numpy as np
from numpy.testing import assert_array_equal
from pathlib import Path
from txrm2tiff.txrm.main import open_txrm
from txrm2tiff.txrm import annot_mixin
from PIL import Image, ImageDraw

from txrm2tiff.xradia_properties.enums import AnnotationTypes, XrmDataTypes


test_data_path = Path("/dls/science/groups/das/ExampleData/B24_test_data/")
test_file = test_data_path / "annotation_test" / "Xray_mosaic_F3C.xrm"

visit_path = test_data_path / "data" / "2019" / "cm98765-1"
raw_path = visit_path / "raw"
xm10_path = raw_path / "XMv10"
xm13_path = raw_path / "XMv13"

test_files = [
    (test_file,),
    (xm13_path / "Xray_mosaic_v13.xrm",),
    (xm13_path / "Xray_mosaic_v13_interrupt.xrm",),
    (xm13_path / "Xray_mosaic_7x7_v13.xrm",),
    (xm13_path / "Xray_single_v13.xrm",),
    (xm13_path / "tomo_v13_full.txrm",),
    (xm13_path / "tomo_v13_full_noref.txrm",),
    (xm13_path / "tomo_v13_interrupt.txrm",),
    (xm13_path / "VLM_mosaic_v13.xrm",),
    (xm13_path / "VLM_mosaic_v13_interrupt.xrm",),
    (xm13_path / "VLM_grid_mosaic_large_v13.xrm",),
]


class TestAnnotator(unittest.TestCase):
    @parameterized.expand(test_files)
    @unittest.skipUnless(visit_path.exists(), "dls paths cannot be accessed")
    def test_with_real_image(self, test_file):
        with open_txrm(test_file) as txrm:
            annotations = txrm.extract_annotations()
        self.assertTrue(np.any(annotations))

    def test_square(self):
        fill = 125
        im_size = 5
        imshape = (1, im_size, im_size)

        x0, y0 = 1, 1
        x1, y1 = 4, 4

        ann = annot_mixin.AnnotatorMixin()
        ann.read_stream = MagicMock()
        ann.output_shape = imshape

        image = Image.new("RGB", imshape[:0:-1], (0, 0, 0))
        draw = ImageDraw.Draw(image, mode="RGB")

        with patch.object(ann, "_get_colour") as patched_colour:
            with patch.object(ann, "_get_thickness") as patched_thickness:
                patched_colour.return_value = (0, fill, 0, 255)
                patched_thickness.return_value = 1
                ann.read_stream.side_effect = [[x0], [y0], [x1], [y1]]
                stream_stem = ""
                ann._plot_rect(draw, stream_stem)

        expected_output = np.zeros((im_size, im_size, 3), dtype=np.uint8)
        y0, y1 = im_size - 1 - y0, im_size - 1 - y1  # Invert y axis
        expected_output[y0, x0 : x1 + 1, 1] = fill
        expected_output[y1 : y0 + 1, x0, 1] = fill
        expected_output[y1, x0 : x1 + 1, 1] = fill
        expected_output[y1 : y0 + 1, x1, 1] = fill

        assert_array_equal(np.asarray(image), expected_output)

    def test_line(self):
        fill = 125
        x0, x1 = 1, 4
        y = 4
        im_size = 5
        imshape = (1, im_size, im_size)
        im = Image.new("RGBA", imshape[:0:-1], (0, 0, 0, 0))
        draw = ImageDraw.Draw(im, mode="RGBA")

        ann = annot_mixin.AnnotatorMixin()
        ann.output_shape = imshape
        ann.read_stream = MagicMock()

        with patch.object(ann, "_get_colour") as patched_colour:
            with patch.object(ann, "_get_thickness") as patched_thickness:
                patched_colour.return_value = (0, fill, 0, 255)
                patched_thickness.return_value = 1
                ann.read_stream.side_effect = [[x0], [x1], [y], [y]]
                ann._plot_line(draw, "")

        output = np.asarray(im.convert("RGB"))

        expected_output = np.zeros((im_size, im_size, 3), dtype=np.uint8)
        y = im_size - 1 - y  # Invert y axis
        expected_output[y, x0 : x1 + 1, 1] = fill

        assert_array_equal(output, expected_output)

    def test_line_and_apply(self):
        fill = (0, 125, 0)
        x0, x1 = 1, 4
        y = 4
        im_size = 5
        im_arr = np.zeros((1, im_size, im_size))
        annotations = Image.new("RGBA", im_arr.shape[:0:-1], (0, 0, 0, 0))
        draw = ImageDraw.Draw(annotations, mode="RGBA")

        ann = annot_mixin.AnnotatorMixin()
        ann.output_shape = im_arr.shape
        ann.read_stream = MagicMock()

        with patch.object(ann, "_get_colour") as patched_colour:
            with patch.object(ann, "_get_thickness") as patched_thickness:
                patched_colour.return_value = (*fill, 255)
                patched_thickness.return_value = 1
                ann.read_stream.side_effect = [[x0], [x1], [y], [y]]
                ann._plot_line(draw, "")

        output = ann.apply_annotations(im_arr, annotations)

        expected_output = np.zeros((1, im_size, im_size, 3), dtype=np.uint8)
        y = im_size - 1 - y  # Invert y axis
        expected_output[0][y, x0 : x1 + 1, :] = fill

        assert_array_equal(output, expected_output)

    def test_ellipse(self):
        fill = 125
        xs = 0, 18
        ys = 2, 16
        x_mid = int(round(np.mean(xs)))
        y_mid = int(round(np.mean(ys)))
        im_size = 20
        imshape = (1, im_size, im_size)
        im = Image.new("RGBA", imshape[:0:-1], (0, 0, 0, 0))
        draw = ImageDraw.Draw(im, mode="RGBA")

        ann = annot_mixin.AnnotatorMixin()
        ann.output_shape = imshape
        ann.read_stream = MagicMock()

        with patch.object(ann, "_get_colour") as patched_colour:
            with patch.object(ann, "_get_thickness") as patched_thickness:
                patched_colour.return_value = (0, fill, 0, 255)
                patched_thickness.return_value = 1
                ann.read_stream.side_effect = [[xs[0]], [ys[0]], [xs[1]], [ys[1]]]
                ann._plot_ellipse(draw, "")

        output = np.asarray(im)

        self.assertTrue((output[:, :, 0] == 0).any())  # Red
        self.assertTrue((output[:, :, 2] == 0).any())  # Blue
        for (x, y) in ann._flip_y(
            (x_mid, ys[0]), (xs[0], y_mid), (x_mid, ys[1]), (xs[1], y_mid)
        ):
            self.assertEqual(
                output[y, x, 1], fill, msg="Failed for y, x of %i, %i" % (y, x)
            )

    def test_polyline(self):
        fill = 125
        im_size = 18  # Must be an even number
        xs = tuple(range(im_size))
        ys = [0, im_size - 1] * (im_size // 2)
        total_points = len(xs)

        imshape = (1, im_size, im_size)
        im = Image.new("RGBA", imshape[:0:-1], (0, 0, 0, 0))
        draw = ImageDraw.Draw(im, mode="RGBA")

        ann = annot_mixin.AnnotatorMixin()
        ann.output_shape = imshape
        ann.read_stream = MagicMock()

        with patch.object(ann, "_get_colour") as patched_colour:
            with patch.object(ann, "_get_thickness") as patched_thickness:
                patched_colour.return_value = (0, fill, 0, 255)
                patched_thickness.return_value = 1
                ann.read_stream.side_effect = [[total_points], xs, ys]
                ann._plot_polyline(draw, "")

        output = np.asarray(im)

        self.assertTrue((output[:, :, 0] == 0).any())  # Red
        self.assertTrue((output[:, :, 2] == 0).any())  # Blue
        for x, y in ann._flip_y(*zip(xs, ys)):
            self.assertEqual(output[y, x, 1], fill)

    def test_scale_bar(self):
        im_size = 5
        imshape = (1, im_size, im_size)
        im = Image.new("RGBA", imshape[:0:-1], (0, 0, 0, 0))
        draw = ImageDraw.Draw(im, mode="RGBA")

        ann = annot_mixin.AnnotatorMixin()
        ann.output_shape = imshape
        ann.image_info = {"PixelSize": [17.0]}
        self.assertTrue(ann._add_scale_bar(draw), msg="Failed to add scale bar")

        output = np.asarray(im)

        self.assertTrue((output[:, :, 0] == 0).any(), msg="You have red on you")  # Red
        self.assertTrue(
            (output[:, :, 1] == 255).any(), msg="You don't have green on you"
        )  # Green
        self.assertTrue(
            (output[:, :, 2] == 0).any(), msg="You have blue on you"
        )  # Blue

    def test_flip_y(self):
        ann = annot_mixin.AnnotatorMixin()
        ann.output_shape = (20, 20)
        res = ann._flip_y((0, 0), (19, 19))
        self.assertEqual(res, ((0, 19), (19, 0)))

    def test_create_image_and_draw(self):
        z, x, y = 2, 6, 5
        channels = 4  # RGBA
        ann = annot_mixin.AnnotatorMixin()
        ann.output_shape = (z, y, x)
        ann.has_stream = MagicMock(return_value=False)
        im, draw = ann._create_image_and_draw()
        self.assertEqual(np.asarray(im).shape, (y, x, channels))

    def test_annotate(self):
        fill = (0, 125, 0)
        x0, x1 = 1, 4
        y = 4
        im = np.zeros((5, 6, 7), dtype=np.uint8)  # Z, Y, X
        num_annotations = 1
        Ann = annot_mixin.AnnotatorMixin
        Ann.thickness_modifier = 1
        ann = Ann()
        ann.get_output = MagicMock(return_value=np.flip(im, axis=1))
        ann.output_shape = im.shape
        ann.has_stream = MagicMock(return_value=True)
        ann.read_stream = MagicMock(
            side_effect=[  # Draw a single line with the listed points
                [num_annotations],
                [AnnotationTypes.ANN_LINE.value],
                [x0],
                [x1],
                [y],
                [y],
            ]
        )
        ann._add_scale_bar = MagicMock(return_value=True)

        with patch.object(ann, "_get_colour") as patched_colour, patch.object(
            ann, "_get_thickness"
        ) as patched_thickness:
            patched_colour.return_value = (*fill, 255)
            patched_thickness.return_value = 1

            ann.annotate()

        ann.read_stream.assert_has_calls(
            [
                call("Annot/TotalAnn", strict=True),
                call("Annot/Ann0/AnnType", XrmDataTypes.XRM_INT, strict=True),
                call("Annot/Ann0/X1", XrmDataTypes.XRM_DOUBLE, strict=True),
                call("Annot/Ann0/X2", XrmDataTypes.XRM_DOUBLE, strict=True),
                call("Annot/Ann0/Y1", XrmDataTypes.XRM_DOUBLE, strict=True),
                call("Annot/Ann0/Y2", XrmDataTypes.XRM_DOUBLE, strict=True),
            ]
        )

        # Extra 3 at the end for RGB
        expected_output = np.zeros((*im.shape, 3), dtype=np.uint8)
        y = expected_output.shape[1] - 1 - y  # Invert y axis
        expected_output[:, y, x0 : x1 + 1, :] = fill
        self.assertEqual(
            ann.annotated_image.shape, expected_output.shape, msg="Shapes don't match"
        )
        unequal_pos = np.where(ann.annotated_image != expected_output)
        assert_array_equal(
            ann.annotated_image,
            expected_output,
            err_msg=f"{[(x, y, z, c) for z, y, x, c in zip(*unequal_pos)]}:\n{ann.annotated_image[unequal_pos]}\n\n{expected_output[unequal_pos]}",
        )
