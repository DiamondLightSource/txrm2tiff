from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

from txrm2tiff.txrm.save_mixin import SaveMixin


class TestSaver(unittest.TestCase):
    @patch("pathlib.Path.mkdir", MagicMock())
    @patch("txrm2tiff.txrm.save_mixin.create_ome_metadata")
    def test_create_metadata(self, mocked_metadata_creator):
        filepath = Path("path/to/file.ext")
        saver = SaveMixin()
        saver.create_metadata(filepath)
        mocked_metadata_creator.assert_called_once_with(saver, filepath.stem)

    @patch.object(SaveMixin, "create_metadata")
    @patch("txrm2tiff.txrm.save_mixin.manual_annotation_save")
    @patch("txrm2tiff.txrm.save_mixin.manual_save")
    @patch("pathlib.Path.mkdir")
    def test_save_image(
        self, mocked_mkdir, mocked_manual_save, mocked_ann_save, mocked_create_metadata
    ):
        filepath = Path("path/to/file.ext")
        dtype = None
        flip = True
        clear_images = True
        mkdir = False

        image = "image"
        metadata = "metadata"

        saver = SaveMixin()

        mocked_create_metadata.return_value = metadata
        saver.get_output = MagicMock(return_value=image)
        saver.referenced = False
        saver.annotated_image = None

        saver.save_image(filepath, dtype, flip, clear_images, mkdir)

        mocked_mkdir.assert_not_called()
        mocked_manual_save.assert_called_once_with(filepath, image, dtype, metadata)
        mocked_ann_save.assert_not_called()

    @patch.object(SaveMixin, "create_metadata")
    @patch("txrm2tiff.txrm.save_mixin.manual_annotation_save")
    @patch("txrm2tiff.txrm.save_mixin.manual_save")
    @patch("pathlib.Path.mkdir")
    def test_save_image_mkdir(
        self, mocked_mkdir, mocked_manual_save, mocked_ann_save, mocked_create_metadata
    ):
        filepath = Path("path/to/file.ext")
        dtype = None
        flip = True
        clear_images = True
        mkdir = True

        image = "image"
        metadata = "metadata"

        saver = SaveMixin()

        mocked_create_metadata.return_value = metadata
        saver.get_output = MagicMock(return_value=image)
        saver.referenced = False
        saver.annotated_image = None

        saver.save_image(filepath, dtype, flip, clear_images, mkdir)

        mocked_mkdir.assert_called_once()
        mocked_manual_save.assert_called_once_with(filepath, image, dtype, metadata)
        mocked_ann_save.assert_not_called()

    @patch.object(SaveMixin, "create_metadata")
    @patch("txrm2tiff.txrm.save_mixin.manual_annotation_save")
    @patch("txrm2tiff.txrm.save_mixin.manual_save")
    @patch("pathlib.Path.mkdir")
    def test_save_image_and_annotations(
        self, mocked_mkdir, mocked_manual_save, mocked_ann_save, mocked_create_metadata
    ):
        filepath = Path("path/to/file.ext")
        dtype = None
        flip = True
        clear_images = True
        mkdir = True

        image = "image"
        metadata = "metadata"
        annotated_image = "annotated"

        saver = SaveMixin()

        mocked_create_metadata.return_value = metadata
        saver.get_output = MagicMock(return_value=image)
        saver.referenced = False
        saver.annotated_image = annotated_image

        saver.save_image(filepath, dtype, flip, clear_images, mkdir)

        mocked_mkdir.assert_called_once()
        mocked_manual_save.assert_called_once_with(filepath, image, dtype, metadata)
        mocked_ann_save.assert_called_once_with(
            Path("path/to/file_Annotated.ext"), annotated_image
        )
