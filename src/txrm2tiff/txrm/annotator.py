from __future__ import annotations
import logging
import math
from pathlib import Path
import typing

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import autocontrast

from ..utils.image_processing import rescale_image
from ..utils.file_handler import manual_annotation_save
from ..xradia_properties import AnnotationTypes, XrmDataTypes as XDTypes

font_path = Path(__file__).parent.parent / "font" / "CallingCode-Regular.otf"

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike
    from numpy.typing import NDArray

    from .abc.images import TxrmWithImages

    TxrmWithImagesType = typing.TypeVar("TxrmWithImagesType", bound=TxrmWithImages)
    FloatOrInt = typing.TypeVar("FloatOrInt", float, int)


class Annotator:

    def __init__(self, txrm: TxrmWithImagesType) -> None:
        self._txrm = txrm
        self.annotated_image: NDArray[np.uint8] | None = None
        self._ann_funcs: dict[
            AnnotationTypes, Callable[[ImageDraw.ImageDraw, str], None]
        ] = {
            AnnotationTypes.ANN_LINE: self._plot_line,
            AnnotationTypes.ANN_RECT: self._plot_rect,
            AnnotationTypes.ANN_ELLIPSE: self._plot_ellipse,
            AnnotationTypes.ANN_CIRCLE: self._plot_ellipse,
            AnnotationTypes.ANN_POLYGON: self._plot_polygon,
            AnnotationTypes.ANN_POLYLINE: self._plot_polyline,
            AnnotationTypes.ANN_FREE_HAND_SKETCH: self._plot_freehand,
        }

    def annotate(
        self,
        scale_bar: bool = True,
        clip_percentiles: tuple[float, float] = (2, 98),
    ) -> NDArray[np.uint8] | None:
        annotations = self.extract_annotations(scale_bar=scale_bar)
        # Checks if anything has been added
        if annotations is None:
            logging.warning("No annotations to apply")
            self.annotated_image = None
        else:
            # Annotations will be in the wrong place if flipped
            images = self._txrm.get_output(flip=False, clear_images=False)
            if images is None:
                logging.warning("No images to annotate")
                return None
            lower_percentile, upper_percentile = np.percentile(images, clip_percentiles)
            images = rescale_image(
                images,
                0,
                255,
                previous_minimum=lower_percentile,
                previous_maximum=upper_percentile,
            )
            self.annotated_image = self.apply_annotations(images, annotations)
        return self.annotated_image

    def get_thickness_modifier(self) -> float:
        """This set a multiplier so that lines are visible when the image is at a sensible size"""
        output_shape = self._txrm.output_shape
        assert output_shape is not None
        return float(np.mean(output_shape[1:]) / 500.0)

    def _create_image_and_draw(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        """Create transparent RGBA image with appropriate 2D dimensions for output image(s)"""
        output_shape = self._txrm.output_shape
        assert output_shape is not None
        im = Image.new("RGBA", output_shape[:0:-1], (0, 0, 0, 0))
        draw = ImageDraw.Draw(im, mode="RGBA")
        return im, draw

    def extract_annotations(self, scale_bar: bool = True) -> Image.Image | None:
        annotations, draw = self._create_image_and_draw()
        annotated = False
        if self._txrm.has_stream("Annot/TotalAnn"):
            num_annotations = int(
                self._txrm.read_stream("Annot/TotalAnn", strict=True)[0]
            )
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

    def apply_annotations(
        self, images: NDArray[typing.Any], annotations: Image.Image
    ) -> NDArray[np.uint8] | None:
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
                image = Image.fromarray(image).convert("L")
                image_auto_contrasted = autocontrast(image)
                image_auto_contrasted_array = np.asarray(image_auto_contrasted)
                if (
                    image_auto_contrasted_array.mean()
                    == image_auto_contrasted_array.max()
                ):
                    image = image_auto_contrasted
                else:
                    del image_auto_contrasted
                image = image.convert("RGBA")
                # Combine annotations and image by alpha
                image.alpha_composite(annotations)
                # Throw away alpha and fill into output array
                output_images[idx] = image.convert("RGB")
            return output_images
        except Exception:
            logging.error("Failed to apply annotations", exc_info=True)
        return None

    def _draw_annotation(self, draw: ImageDraw.ImageDraw, index: int) -> None:
        stream_stem = "Annot/Ann%i" % index
        ann_type_int = int(
            self._txrm.read_stream(
                f"{stream_stem}/AnnType", XDTypes.XRM_INT, strict=True
            )[0]
        )
        self._ann_funcs.get(AnnotationTypes(ann_type_int), self._not_implemented)(
            draw, stream_stem
        )

    def _add_scale_bar(self, draw: ImageDraw.ImageDraw) -> bool:
        pixel_size = self._txrm.image_info["PixelSize"][0]  # microns
        if pixel_size is not None and pixel_size > 0:
            try:
                colour = (0, 255, 0, 255)  # Solid green
                # Set up scale bar dims:
                assert self._txrm.output_shape is not None
                x0, y0 = self._flip_y(
                    (
                        self._txrm.output_shape[2] // 50,
                        self._txrm.output_shape[1] // 30,
                    ),
                    output_shape=self._txrm.output_shape,
                )[0]
                thickness_modifier = self.get_thickness_modifier()
                bar_width = int(6 * thickness_modifier)
                # Set up scale bar text:
                f = ImageFont.truetype(
                    str(font_path),
                    int(15 * thickness_modifier),
                )

                tmp_bar_length = (
                    self._txrm.output_shape[2] / 5.0
                )  # Get initial bar length based on image width
                tmp_bar_size = tmp_bar_length * pixel_size  # Calculate physical size

                # Round to nearest multiple of the order of magnitude
                exponent = math.floor(math.log10(tmp_bar_size))
                step_size = 10.0**exponent
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
                text_xy = (x0 + bar_length / 2, y0 - bar_width)
                # Draw:
                draw.text(
                    text_xy, text, font=f, fill=colour, anchor="mb"
                )  # anchor: middle-bottom
                draw.line((x0, y0, x0 + bar_length, y0), fill=colour, width=bar_width)
                return True
            except Exception:
                logging.error(
                    "Exception occurred while drawing scale bar", exc_info=True
                )
        return False

    def _get_thickness(self, stream_stem: str) -> int:
        return int(
            float(
                self._txrm.read_stream(
                    f"{stream_stem}/AnnStrokeThickness", XDTypes.XRM_DOUBLE, strict=True
                )[0]
            )
            * self.get_thickness_modifier()
        )

    def _get_colour(self, stream_stem: str) -> tuple[int, int, int, int]:
        colours = self._txrm.read_stream(
            f"{stream_stem}/AnnColor", XDTypes.XRM_UNSIGNED_CHAR
        )
        return (colours[2], colours[1], colours[0], colours[3])

    @staticmethod
    def _not_implemented(*args: typing.Any) -> None:
        """Do nothing"""
        pass

    def _plot_line(self, draw: ImageDraw.ImageDraw, stream_stem: str) -> None:
        colour = self._get_colour(stream_stem)
        thickness = self._get_thickness(stream_stem)
        x1 = self._txrm.read_stream(
            f"{stream_stem}/X1", XDTypes.XRM_FLOAT, strict=True
        )[0]
        x2 = self._txrm.read_stream(
            f"{stream_stem}/X2", XDTypes.XRM_FLOAT, strict=True
        )[0]
        y1 = self._txrm.read_stream(
            f"{stream_stem}/Y1", XDTypes.XRM_FLOAT, strict=True
        )[0]
        y2 = self._txrm.read_stream(
            f"{stream_stem}/Y2", XDTypes.XRM_FLOAT, strict=True
        )[0]

        assert self._txrm.output_shape is not None
        draw.line(
            self._flip_y((x1, y1), (x2, y2), output_shape=self._txrm.output_shape),
            fill=colour,
            width=thickness,
        )

    def __extract_rectangle_coords(
        self, stream_stem: str
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        x = []
        y = []
        x.append(
            self._txrm.read_stream(
                f"{stream_stem}/Rectangle/Left", XDTypes.XRM_INT, strict=True
            )[0]
        )
        y.append(
            self._txrm.read_stream(
                f"{stream_stem}/Rectangle/Bottom", XDTypes.XRM_INT, strict=True
            )[0]
        )
        x.append(
            self._txrm.read_stream(
                f"{stream_stem}/Rectangle/Right", XDTypes.XRM_INT, strict=True
            )[0]
        )
        y.append(
            self._txrm.read_stream(
                f"{stream_stem}/Rectangle/Top", XDTypes.XRM_INT, strict=True
            )[0]
        )
        x.sort()
        y.sort(reverse=True)
        # Values in the first tuple must be smaller than their respective second tuple value.
        # Therefore y1 must be first and y0 second because they are flipped.
        assert self._txrm.output_shape is not None
        coords = self._flip_y(*zip(x, y), output_shape=self._txrm.output_shape)
        return (coords[0], coords[1])

    def _plot_rect(self, draw: ImageDraw.ImageDraw, stream_stem: str) -> None:
        colour = self._get_colour(stream_stem)
        thickness = self._get_thickness(stream_stem)
        xy = self.__extract_rectangle_coords(stream_stem)

        draw.rectangle(xy, outline=colour, width=thickness)

    def _plot_ellipse(self, draw: ImageDraw.ImageDraw, stream_stem: str) -> None:
        colour = self._get_colour(stream_stem)
        thickness = self._get_thickness(stream_stem)
        xy = self.__extract_rectangle_coords(stream_stem)

        draw.ellipse(xy, outline=colour, width=thickness)

    def _plot_polygon(self, draw: ImageDraw.ImageDraw, stream_stem: str) -> None:
        colour = self._get_colour(stream_stem)
        thickness = self._get_thickness(stream_stem)
        total_points = self._txrm.read_stream(
            f"{stream_stem}/TotalPts", XDTypes.XRM_UNSIGNED_INT
        )[0]

        xs = self._txrm.read_stream(
            f"{stream_stem}/PointX", XDTypes.XRM_FLOAT, strict=True
        )[:total_points]
        ys = self._txrm.read_stream(
            f"{stream_stem}/PointY", XDTypes.XRM_FLOAT, strict=True
        )[:total_points]
        xs.append(xs[0])  # Link beginning and end points
        ys.append(ys[0])

        output_shape = self._txrm.output_shape
        assert output_shape is not None
        draw.line(
            self._flip_y(*zip(xs, ys), output_shape=output_shape),
            fill=colour,
            width=thickness,
        )

    def _plot_polyline(
        self,
        draw: ImageDraw.ImageDraw,
        stream_stem: str,
        joint: typing.Literal["curve"] | None = None,
    ) -> None:
        colour = self._get_colour(stream_stem)
        thickness = self._get_thickness(stream_stem)
        total_points = self._txrm.read_stream(
            f"{stream_stem}/TotalPts", XDTypes.XRM_UNSIGNED_INT
        )[0]
        xs = self._txrm.read_stream(
            f"{stream_stem}/PointX", XDTypes.XRM_FLOAT, strict=True
        )
        ys = self._txrm.read_stream(
            f"{stream_stem}/PointY", XDTypes.XRM_FLOAT, strict=True
        )

        output_shape = self._txrm.output_shape
        assert output_shape is not None
        draw.line(
            self._flip_y(
                *zip(xs[:total_points], ys[:total_points]),
                output_shape=output_shape,
            ),
            fill=colour,
            width=thickness,
            joint=joint,
        )

    def _plot_freehand(self, draw: ImageDraw.ImageDraw, stream_stem: str) -> None:
        self._plot_polyline(draw, stream_stem, joint="curve")

    def _flip_y(
        self, *xys: tuple[FloatOrInt, FloatOrInt], output_shape: tuple[int, int, int]
    ) -> tuple[tuple[FloatOrInt, FloatOrInt], ...]:
        """
        Stored x-y coordinates are assuming 0 is bottom left, whereas PIL assumes 0 is top left
        This flips any list of coordinates that alternate x, y

        Args:
            *xys: n args assuming the form (x0, y0), (x1, y1), ... (xn, yn)

        Returns:
            Tuple of tuples: x-y coordinates with a flipped y
        """
        return tuple((x, output_shape[1] - 1 - y) for x, y in xys)

    def save(
        self,
        filepath: str | PathLike[str],
        mkdir: bool = False,
        strict: bool = True,
    ) -> bool:
        """Saves images (if available) returning True if successful."""
        try:
            filepath = Path(filepath)
            if self.annotated_image is None:
                raise AttributeError("No annotated image to save")
            if mkdir:
                filepath.parent.mkdir(parents=True, exist_ok=True)

            manual_annotation_save(
                filepath,
                self.annotated_image,
            )
            return True
        except Exception:
            logging.error("Saving failed", exc_info=not strict)
            if strict:
                raise
            return False
