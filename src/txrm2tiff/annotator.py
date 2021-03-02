import logging
from pathlib import Path
from math import ceil
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import autocontrast

from . import txrm_wrapper

font_path = Path(__file__).parent / "font" / "CallingCode-Regular.otf"


class Annotator:
    
    def __init__(self, xy_dims):
        """
        Sets up Annotator class for drawing and saving annotations over a converted image.

        Args:
            xy_dims: Tuple or list of dimensions of the image to annotate in the form (x, y)
        """
        self.ann_types = defaultdict(lambda: self._not_implemented, {
            0:  self._plot_line,        # ANN_LINE
            1:  self._plot_rect,        # ANN_RECT
            2:  self._plot_ellipse,     # ANN_ELLIPSE
            3:  self._plot_ellipse,     # ANN_CIRCLE
            # 4:  self._not_implemented,  # ANN_TEXT (Doesn't work in XRMDataExplorer)
            5:  self._plot_polygon,     # ANN_POLYGON
            # 6:  self._not_implemented,  # ANN_GROUP
            # 7:  self._not_implemented,  # ANN_LINE_PLOT
            # 8:  self._not_implemented,  # ANN_RULE
            # 9:  self._not_implemented,  # ANN_ANGLE
            # 10: self._not_implemented,  # ANN_MOUSEPOINT
            # 11: self._not_implemented,  # ANN_CONE_ANGLE_LINE
            # 12: self._not_implemented,  # ANN_HIGHLIGHTER
            13: self._plot_polyline,    # ANN_POLYLINE
            14: self._plot_freehand,    # ANN_FREE_HAND_SKETCH
            # 15: self._not_implemented,  # ANN_SIZE
        })

        self._saved_annotations = False

        self._xy = xy_dims  # Output image dims

        # This set a multiplier so that lines are visible when the image is at a sensible size 
        self._thickness_multiplier = np.mean(self._xy) / 500

        # Create transparent RGBA image with input dims
        self._annotations = Image.new("RGBA", self._xy, (0, 0, 0, 0))
        self._draw = ImageDraw.Draw(self._annotations, mode="RGBA")

    def extract_annotations(self, ole):
        """
        Draws annotations over an RGB copy of the converted image using PIL.

        Args:
            ole: An instance of the xrm/txrm file opened by olefile

        Returns:
            [bool]: True if any annotations have been successfully drawn, otherwise False.
        """
        if ole.exists("Annot/TotalAnn"):
            num_annotations = txrm_wrapper.read_stream(ole, "Annot/TotalAnn", np.intc)[0]
            if num_annotations > 0:
                for i in range(num_annotations):
                    stream_stem = "Annot/Ann%i" %i
                    try:
                        ann_type = txrm_wrapper.read_stream(ole, f"{stream_stem}/AnnType", np.intc)[0]
                        self.ann_types.get(ann_type)(ole, stream_stem)
                        self._saved_annotations = True
                    except Exception:
                        # Create error message but don't stop other annotations from being extracted
                        logging.error("Failed to get annotation %i", i, exc_info=True)
        # _add_scale_bar() returns True if scale bar successfully added
        self._saved_annotations = self._add_scale_bar(ole) or self._saved_annotations
        return self._saved_annotations

    def apply_annotations(self, images):
        """
        Apply annotations to images returning a numpy.ndarray

        Args:
            images (3D NumPy array): Images that will be annotated

        Returns:
            RGB numpy array or None: Annotated images, if successfully applied
        """
        # Create output array with shape of input, plus 3 channels for RGB
        output_images = np.zeros((*images.shape, 3), dtype="uint8")
        try:
            if not self._saved_annotations:
                raise AttributeError("Annotations cannot be as no annotations were successfully extracted")
            images = np.flip(images, 1)
            for idx, image in enumerate(images, 0):
                # Make 'L' type PIL image from 2D array, autocontrast, then convert to RGBA 
                image = autocontrast(Image.fromarray(image).convert("L")).convert("RGBA")
                # Combine annotations and image by alpha
                image.alpha_composite(self._annotations)
                # Throw away alpha and fill into output array
                output_images[idx] = image.convert("RGB")
            return output_images
        except Exception:
            logging.error("Failed to apply annotations", exc_info=True)

    def _add_scale_bar(self, ole):
        pixel_size = txrm_wrapper.extract_pixel_size(ole) # microns
        if pixel_size is not None and pixel_size > 0:
            try:
                colour = (0, 255, 0, 255) # Solid green
                # Set up scale bar dims:
                x0, y0 = self._flip_y((self._xy[0] // 50, self._xy[1] // 30))[0]
                bar_length = ceil(self._xy[0] / 4.)
                bar_width = int(6 * self._thickness_multiplier)
                # Set up scale bar text:
                f = ImageFont.truetype(str(font_path), int(15 * self._thickness_multiplier))
                text_len = f.size
                text_height = f.font.height
                text = "%iμm" %(bar_length * pixel_size)
                text_x = x0 + bar_length // 2 - text_len
                text_y = y0 - bar_width - text_height
                # Draw:
                self._draw.text((text_x, text_y), text, font=f, fill=colour)
                self._draw.line((x0, y0, x0 + bar_length, y0), fill=colour, width=bar_width)
                return True
            except Exception:
                logging.error("Exception occurred while drawing scale bar", exc_info=True)
        return False

    def _get_thickness(self, ole, stream_stem):
        return int(txrm_wrapper.read_stream(ole, f"{stream_stem}/AnnStrokeThickness", np.double)[0] * self._thickness_multiplier)

    @staticmethod
    def _get_colour(ole, stream_stem):
        colours = txrm_wrapper.read_stream(ole, f"{stream_stem}/AnnColor", np.ubyte)
        return (*colours[2::-1], colours[3])

    @staticmethod
    def _not_implemented(*args):
        pass

    def _plot_line(self, ole, stream_stem):
        colour = self._get_colour(ole, stream_stem)
        thickness = self._get_thickness(ole, stream_stem)
        x1 = txrm_wrapper.read_stream(ole, f"{stream_stem}/X1", np.single)[0]
        x2 = txrm_wrapper.read_stream(ole, f"{stream_stem}/X2", np.single)[0]
        y1 = txrm_wrapper.read_stream(ole, f"{stream_stem}/Y1", np.single)[0]
        y2 = txrm_wrapper.read_stream(ole, f"{stream_stem}/Y2", np.single)[0]

        self._draw.line(self._flip_y((x1, y1), (x2, y2)), fill=colour, width=thickness)

    def _plot_rect(self, ole, stream_stem):
        colour = self._get_colour(ole, stream_stem)
        thickness = self._get_thickness(ole, stream_stem)
        x0 = txrm_wrapper.read_stream(ole, f"{stream_stem}/Rectangle/Left", np.intc)[0]
        y0 = txrm_wrapper.read_stream(ole, f"{stream_stem}/Rectangle/Bottom", np.intc)[0]
        x1 = txrm_wrapper.read_stream(ole, f"{stream_stem}/Rectangle/Right", np.intc)[0]
        y1 = txrm_wrapper.read_stream(ole, f"{stream_stem}/Rectangle/Top", np.intc)[0]

        self._draw.rectangle(self._flip_y((x0, y0), (x1, y1)), outline=colour, width=thickness)

    def _plot_ellipse(self, ole, stream_stem):
        colour = self._get_colour(ole, stream_stem)
        thickness = self._get_thickness(ole, stream_stem)
        x0 = txrm_wrapper.read_stream(ole, f"{stream_stem}/Rectangle/Left", np.intc)[0]
        y0 = txrm_wrapper.read_stream(ole, f"{stream_stem}/Rectangle/Bottom", np.intc)[0]
        x1 = txrm_wrapper.read_stream(ole, f"{stream_stem}/Rectangle/Right", np.intc)[0]
        y1 = txrm_wrapper.read_stream(ole, f"{stream_stem}/Rectangle/Top", np.intc)[0]
        # Values in the first tuple must be smaller than their respective second tuple value.
        # Therefore y1 must be first and y0 second because they are flipped.
        self._draw.ellipse(self._flip_y((x0, y1), (x1, y0)), outline=colour, width=thickness)

    def _plot_polygon(self, ole, stream_stem):
        colour = self._get_colour(ole, stream_stem)
        thickness = self._get_thickness(ole, stream_stem)
        total_points = txrm_wrapper.read_stream(ole, f"{stream_stem}/TotalPts", np.uintc)[0]
        xs = txrm_wrapper.read_stream(ole, f"{stream_stem}/PointX", np.single)[:total_points]
        ys = txrm_wrapper.read_stream(ole, f"{stream_stem}/PointY", np.single)[:total_points]
        xs.append(xs[0])  # Link beginning and end points
        ys.append(ys[0])
        
        self._draw.line(self._flip_y(*zip(xs, ys)), fill=colour, width=thickness)

    def _plot_polyline(self, ole, stream_stem, joint=None):
        colour = self._get_colour(ole, stream_stem)
        thickness = self._get_thickness(ole, stream_stem)
        total_points = txrm_wrapper.read_stream(ole, f"{stream_stem}/TotalPts", np.uintc)[0]
        xs = txrm_wrapper.read_stream(ole, f"{stream_stem}/PointX", np.single)
        ys = txrm_wrapper.read_stream(ole, f"{stream_stem}/PointY", np.single)

        self._draw.line(self._flip_y(*zip(xs[:total_points], ys[:total_points])), fill=colour, width=thickness, joint=joint)

    def _plot_freehand(self, ole, stream_stem):
        self._plot_polyline(ole, stream_stem, joint="curve")

    def _flip_y(self, *xys):
        """
        Stored x-y coordinates are assuming 0 is bottom left, whereas PIL assumes 0 is top left
        This flips any list of coordinates that alternate x, y

        Args:
            *xys: n args assuming the form (x0, y0), (x1, y1), ... (xn, yn)

        Returns:
            tuple of tuples: x-y coordinates with a flipped y
        """
        xys = tuple((x, self._xy[1] - 1 - y) for (x, y) in xys)
        return xys
