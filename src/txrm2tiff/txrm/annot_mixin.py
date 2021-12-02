import logging
import math
from numbers import Number
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import autocontrast

from ..xradia_properties import AnnotationTypes, XrmDataTypes as XDT

font_path = Path(__file__).parent.parent / "font" / "CallingCode-Regular.otf"


class AnnotatorMixin:
    def __new__(cls, *args, **kwargs):
        cls.setup_ann_functions()
        return super().__new__(cls)

    @classmethod
    def setup_ann_functions(cls):
        cls.ann_funcs = defaultdict(
            lambda: cls._not_implemented,
            {
                AnnotationTypes.ANN_LINE: cls._plot_line,
                AnnotationTypes.ANN_RECT: cls._plot_rect,
                AnnotationTypes.ANN_ELLIPSE: cls._plot_ellipse,
                AnnotationTypes.ANN_CIRCLE: cls._plot_ellipse,
                AnnotationTypes.ANN_POLYGON: cls._plot_polygon,
                AnnotationTypes.ANN_POLYLINE: cls._plot_polyline,
                AnnotationTypes.ANN_FREE_HAND_SKETCH: cls._plot_freehand,
            },
        )

    def annotate(self) -> Optional[np.ndarray]:
        """Annotate output image. Please ensure that the image has been referenced first, if applicable."""
        annotations = self.extract_annotations(scale_bar=True)
        # Checks if anything has been added
        if annotations is None:
            logging.warning("No annotations to apply")
            self.annotated_image = None
        else:
            # Annotations will be in the wrong place if flipped
            images = self.get_output(flip=False, clear_images=False)
            self.annotated_image = self.apply_annotations(images, annotations)
        return self.annotated_image

    @property
    def thickness_modifier(self) -> float:
        """This set a multiplier so that lines are visible when the image is at a sensible size"""
        if not hasattr(self, "output_shape"):
            return 1.0
        return np.mean(self.output_shape[0]) / 500.0

    def _create_image_and_draw(self):
        """Create transparent RGBA image with appropriate 2D dimensions for output image(s)"""
        im = Image.new("RGBA", self.output_shape[:0:-1], (0, 0, 0, 0))
        draw = ImageDraw.Draw(im, mode="RGBA")
        return im, draw

    def extract_annotations(self, scale_bar: bool = True) -> Optional[np.ndarray]:
        annotations, draw = self._create_image_and_draw()
        annotated = False
        if self.has_stream("Annot/TotalAnn"):
            num_annotations = self.read_stream("Annot/TotalAnn", strict=True)[0]
            if num_annotations is not None and num_annotations > 0:
                for i in range(num_annotations):
                    try:
                        self._draw_annotation(draw, i)
                        annotated = True
                    except Exception:
                        # Create error message but don't stop other annotations from being extracted
                        logging.error("Failed to get annotation %i", i, exc_info=True)
        # _add_scale_bar() returns True if scale bar successfully added
        if scale_bar:
            annotated = self._add_scale_bar(draw) or annotated
        if not annotated:
            logging.warning("No annotations were extracted")
            return None
        return annotations

    def apply_annotations(self, images: np.ndarray, annotations: Image) -> np.ndarray:
        """
        Apply annotations to images returning a numpy.ndarray

        Args:
            images (3D NumPy array): Images that will be annotated

        Returns:
            RGB numpy array or None: Annotated images, if successfully applied
        """
        try:
            # Create output array with shape of input, plus 3 channels for RGB
            output_images = np.zeros((*images.shape, 3), dtype="uint8")
            for idx, image in enumerate(images, 0):
                # Make 'L' type PIL image from 2D array, autocontrast, then convert to RGBA
                image = autocontrast(Image.fromarray(image).convert("L")).convert(
                    "RGBA"
                )
                # Combine annotations and image by alpha
                image.alpha_composite(annotations)
                # Throw away alpha and fill into output array
                output_images[idx] = image.convert("RGB")
            return output_images
        except Exception:
            logging.error("Failed to apply annotations", exc_info=True)

    def _draw_annotation(self, draw: ImageDraw, index: int) -> None:
        stream_stem = "Annot/Ann%i" % index
        ann_type_int = self.read_stream(
            f"{stream_stem}/AnnType", XDT.XRM_INT, strict=True
        )[0]
        self.ann_funcs.get(AnnotationTypes(ann_type_int))(self, draw, stream_stem)

    def _add_scale_bar(self, draw: ImageDraw) -> bool:
        pixel_size = self.image_info["PixelSize"][0]  # microns
        if pixel_size is not None and pixel_size > 0:
            try:
                colour = (0, 255, 0, 255)  # Solid green
                # Set up scale bar dims:
                x0, y0 = self._flip_y(
                    (self.output_shape[2] // 50, self.output_shape[1] // 30)
                )[0]
                bar_width = int(6 * self.thickness_modifier)
                # Set up scale bar text:
                f = ImageFont.truetype(
                    str(font_path), int(15 * self.thickness_modifier)
                )

                tmp_bar_length = (
                    self.output_shape[2] / 5.0
                )  # Get initial bar length based on image width
                tmp_bar_size = tmp_bar_length * pixel_size  # Calculate physical size

                # Round to nearest multiple of the order of magnitude
                exponent = math.floor(math.log10(tmp_bar_size))
                step_size = 10.0 ** exponent
                bar_size = round(
                    round(tmp_bar_size / step_size)
                    * step_size,  # Find nearest multiple of step_size
                    -exponent,
                )  # round to -exponent decimal places (to mitigate rounding errors)
                bar_length = (
                    bar_size / pixel_size
                )  # Calculate number of pixels that wide new bar size bar should be

                # Set text and calculate text positions:
                text = "%gμm" % bar_size
                text_width, text_height = draw.textsize(text, font=f)
                text_x = round(
                    x0 + bar_length / 2 - (text_width / 2)
                )  # Centre text above bar
                text_y = y0 - bar_width - text_height
                # Draw:
                draw.text((text_x, text_y), text, font=f, fill=colour)
                draw.line((x0, y0, x0 + bar_length, y0), fill=colour, width=bar_width)
                return True
            except Exception:
                logging.error(
                    "Exception occurred while drawing scale bar", exc_info=True
                )
        return False

    def _get_thickness(self, stream_stem: str) -> int:
        return int(
            self.read_stream(
                f"{stream_stem}/AnnStrokeThickness", XDT.XRM_DOUBLE, strict=True
            )[0]
            * self.thickness_modifier
        )

    def _get_colour(self, stream_stem: str) -> Tuple:
        colours = self.read_stream(f"{stream_stem}/AnnColor", XDT.XRM_UNSIGNED_CHAR)
        return (*colours[2::-1], colours[3])

    @staticmethod
    def _not_implemented(*args):
        """Do nothing"""
        pass

    def _plot_line(self, draw: ImageDraw, stream_stem: str):
        colour = self._get_colour(stream_stem)
        thickness = self._get_thickness(stream_stem)
        x1 = self.read_stream(f"{stream_stem}/X1", XDT.XRM_DOUBLE, strict=True)[0]
        x2 = self.read_stream(f"{stream_stem}/X2", XDT.XRM_DOUBLE, strict=True)[0]
        y1 = self.read_stream(f"{stream_stem}/Y1", XDT.XRM_DOUBLE, strict=True)[0]
        y2 = self.read_stream(f"{stream_stem}/Y2", XDT.XRM_DOUBLE, strict=True)[0]

        draw.line(self._flip_y((x1, y1), (x2, y2)), fill=colour, width=thickness)

    def _plot_rect(self, draw: ImageDraw, stream_stem: str):
        colour = self._get_colour(stream_stem)
        thickness = self._get_thickness(stream_stem)
        x0 = self.read_stream(
            f"{stream_stem}/Rectangle/Left", XDT.XRM_INT, strict=True
        )[0]
        y0 = self.read_stream(
            f"{stream_stem}/Rectangle/Bottom", XDT.XRM_INT, strict=True
        )[0]
        x1 = self.read_stream(
            f"{stream_stem}/Rectangle/Right", XDT.XRM_INT, strict=True
        )[0]
        y1 = self.read_stream(f"{stream_stem}/Rectangle/Top", XDT.XRM_INT, strict=True)[
            0
        ]

        draw.rectangle(
            self._flip_y((x0, y0), (x1, y1)), outline=colour, width=thickness
        )

    def _plot_ellipse(self, draw: ImageDraw, stream_stem: str):
        colour = self._get_colour(stream_stem)
        thickness = self._get_thickness(stream_stem)
        x0 = self.read_stream(
            f"{stream_stem}/Rectangle/Left", XDT.XRM_INT, strict=True
        )[0]
        y0 = self.read_stream(
            f"{stream_stem}/Rectangle/Bottom", XDT.XRM_INT, strict=True
        )[0]
        x1 = self.read_stream(
            f"{stream_stem}/Rectangle/Right", XDT.XRM_INT, strict=True
        )[0]
        y1 = self.read_stream(f"{stream_stem}/Rectangle/Top", XDT.XRM_INT, strict=True)[
            0
        ]
        # Values in the first tuple must be smaller than their respective second tuple value.
        # Therefore y1 must be first and y0 second because they are flipped.
        draw.ellipse(self._flip_y((x0, y1), (x1, y0)), outline=colour, width=thickness)

    def _plot_polygon(self, draw: ImageDraw, stream_stem: str) -> None:
        colour = self._get_colour(stream_stem)
        thickness = self._get_thickness(stream_stem)
        total_points = self.read_stream(f"{stream_stem}/TotalPts", np.uintc)[0]
        xs = self.read_stream(f"{stream_stem}/PointX", XDT.XRM_DOUBLE, strict=True)[
            :total_points
        ]
        ys = self.read_stream(f"{stream_stem}/PointY", XDT.XRM_DOUBLE, strict=True)[
            :total_points
        ]
        xs.append(xs[0])  # Link beginning and end points
        ys.append(ys[0])

        draw.line(self._flip_y(*zip(xs, ys)), fill=colour, width=thickness)

    def _plot_polyline(
        self, draw: ImageDraw, stream_stem: str, joint: Optional[str] = None
    ):
        colour = self._get_colour(stream_stem)
        thickness = self._get_thickness(stream_stem)
        total_points = self.read_stream(f"{stream_stem}/TotalPts", np.uintc)[0]
        xs = self.read_stream(f"{stream_stem}/PointX", XDT.XRM_DOUBLE, strict=True)
        ys = self.read_stream(f"{stream_stem}/PointY", XDT.XRM_DOUBLE, strict=True)

        draw.line(
            self._flip_y(*zip(xs[:total_points], ys[:total_points])),
            fill=colour,
            width=thickness,
            joint=joint,
        )

    def _plot_freehand(self, draw: ImageDraw, stream_stem: str) -> None:
        self._plot_polyline(draw, stream_stem, joint="curve")

    def _flip_y(self, *xys: Iterable[Number]) -> Tuple[Tuple[Number, Number]]:
        """
        Stored x-y coordinates are assuming 0 is bottom left, whereas PIL assumes 0 is top left
        This flips any list of coordinates that alternate x, y

        Args:
            *xys: n args assuming the form (x0, y0), (x1, y1), ... (xn, yn)

        Returns:
            tuple of tuples: x-y coordinates with a flipped y
        """
        xys = tuple((x, self.output_shape[1] - 1 - y) for (x, y) in xys)
        return xys
