from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock, call

from txrm2tiff.txrm.save_mixin import SaveMixin
from txrm2tiff.txrm.annot_mixin import AnnotatorMixin


class SaveAnnotatorMixin(SaveMixin, AnnotatorMixin):
    ...


class TestSaver(unittest.TestCase):
    @patch("pathlib.Path.mkdir", MagicMock())
    @patch("txrm2tiff.txrm.save_mixin.create_ome_metadata")
    def test_create_metadata(self, mocked_metadata_creator):
        filepath = Path("path/to/file.ext")
        saver = SaveMixin()
        saver.create_metadata(filepath)
        mocked_metadata_creator.assert_called_once_with(saver, filepath.stem)

    @patch.object(SaveMixin, "create_metadata")
    @patch("txrm2tiff.txrm.annot_mixin.manual_annotation_save")
    @patch("txrm2tiff.txrm.save_mixin.manual_save")
    @patch("pathlib.Path.mkdir")
    def test_save_images(
        self, mocked_mkdir, mocked_manual_save, mocked_ann_save, mocked_create_metadata
    ):
        filepath = Path("path/to/file.ext")
        dtype = None
        shifts = True
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

        self.assertTrue(
            saver.save_images(
                filepath, dtype, shifts, flip, clear_images, mkdir, strict=True
            )
        )

        saver.get_output.assert_called_once_with(
            load=True, shifts=shifts, flip=flip, clear_images=clear_images
        )
        mocked_mkdir.assert_not_called()
        mocked_manual_save.assert_called_once_with(filepath, image, dtype, metadata)
        mocked_ann_save.assert_not_called()

    @patch.object(SaveMixin, "create_metadata")
    @patch("txrm2tiff.txrm.save_mixin.manual_save")
    @patch("pathlib.Path.mkdir")
    def test_save_images_mkdir(
        self, mocked_mkdir, mocked_manual_save, mocked_create_metadata
    ):
        filepath = Path("path/to/file.ext")
        dtype = None
        shifts = True
        flip = True
        clear_images = True
        mkdir = True

        image = "image"
        metadata = "metadata"

        saver = SaveMixin()

        mocked_create_metadata.return_value = metadata
        saver.get_output = MagicMock(return_value=image)
        saver.save_annotations = MagicMock()
        saver.referenced = False
        saver.annotated_image = None

        self.assertTrue(
            saver.save_images(
                filepath, dtype, shifts, flip, clear_images, mkdir, strict=True
            )
        )

        saver.get_output.assert_called_once_with(
            load=True, shifts=shifts, flip=flip, clear_images=clear_images
        )
        mocked_mkdir.assert_called_once()
        mocked_manual_save.assert_called_once_with(filepath, image, dtype, metadata)
        saver.save_annotations.assert_not_called()

    @patch.object(SaveMixin, "create_metadata")
    @patch("txrm2tiff.txrm.save_mixin.manual_save")
    @patch("pathlib.Path.mkdir")
    def test_save_images_creates_ome_tiff(
        self, mocked_mkdir, mocked_manual_save, mocked_create_metadata
    ):
        filepath = Path("path/to/file.ext")
        dtype = None
        shifts = True
        flip = True
        clear_images = True
        mkdir = False

        image = "image"
        metadata = "metadata"

        saver = SaveMixin()

        mocked_create_metadata.return_value = metadata
        saver.path = filepath
        saver.get_output = MagicMock(return_value=image)
        saver.save_annotations = MagicMock()
        saver.referenced = False
        saver.annotated_image = None

        self.assertTrue(
            saver.save_images(
                None, dtype, shifts, flip, clear_images, mkdir, strict=True
            )
        )

        saver.get_output.assert_called_once_with(
            load=True, shifts=shifts, flip=flip, clear_images=clear_images
        )
        mocked_mkdir.assert_not_called()
        mocked_manual_save.assert_called_once_with(
            filepath.resolve().with_suffix(".ome.tiff"), image, dtype, metadata
        )
        saver.save_annotations.assert_not_called()

    @patch.object(SaveMixin, "create_metadata")
    @patch("txrm2tiff.txrm.annot_mixin.manual_annotation_save")
    @patch("txrm2tiff.txrm.save_mixin.manual_save")
    @patch("pathlib.Path.mkdir")
    def test_save_images_and_annotations_simple_ext(
        self, mocked_mkdir, mocked_manual_save, mocked_ann_save, mocked_create_metadata
    ):
        filepath = Path("path/to/file.ext")
        dtype = None
        shifts = False
        flip = True
        clear_images = True
        mkdir = True

        image = "image"
        metadata = "metadata"
        annotated_image = "annotated"

        saver = SaveAnnotatorMixin()

        mocked_create_metadata.return_value = metadata
        saver.get_output = MagicMock(return_value=image)
        saver.referenced = False
        saver.annotated_image = annotated_image

        self.assertTrue(
            saver.save_images(
                filepath, dtype, shifts, flip, clear_images, mkdir, strict=True
            )
        )

        saver.get_output.assert_called_once_with(
            load=True, shifts=shifts, flip=flip, clear_images=clear_images
        )
        mocked_mkdir.assert_has_calls([call(parents=True, exist_ok=True)] * 2)
        mocked_manual_save.assert_called_once_with(filepath, image, dtype, metadata)
        mocked_ann_save.assert_called_once_with(
            Path("path/to/file_Annotated.ext"), annotated_image
        )

    @patch.object(SaveMixin, "create_metadata")
    @patch("txrm2tiff.txrm.annot_mixin.manual_annotation_save")
    @patch("txrm2tiff.txrm.save_mixin.manual_save")
    @patch("pathlib.Path.mkdir")
    def test_save_images_and_annotations_tiff(
        self, mocked_mkdir, mocked_manual_save, mocked_ann_save, mocked_create_metadata
    ):
        filepath = Path("path/to/file.ome.tiff")
        dtype = None
        shifts = False
        flip = True
        clear_images = True
        mkdir = False

        image = "image"
        metadata = "metadata"
        annotated_image = "annotated"

        saver = SaveAnnotatorMixin()

        mocked_create_metadata.return_value = metadata
        saver.get_output = MagicMock(return_value=image)
        saver.referenced = False
        saver.annotated_image = annotated_image

        self.assertTrue(
            saver.save_images(
                filepath, dtype, shifts, flip, clear_images, mkdir, strict=True
            )
        )

        saver.get_output.assert_called_once_with(
            load=True, shifts=shifts, flip=flip, clear_images=clear_images
        )
        mocked_mkdir.assert_not_called()
        mocked_manual_save.assert_called_once_with(filepath, image, dtype, metadata)
        mocked_ann_save.assert_called_once_with(
            Path("path/to/file_Annotated.tiff"), annotated_image
        )

    @patch.object(SaveMixin, "create_metadata")
    @patch("txrm2tiff.txrm.annot_mixin.manual_annotation_save")
    @patch("txrm2tiff.txrm.save_mixin.manual_save")
    @patch("pathlib.Path.mkdir")
    def test_save_images_and_annotations_specific_filepath(
        self, mocked_mkdir, mocked_manual_save, mocked_ann_save, mocked_create_metadata
    ):
        filepath = Path("path/to/file.ome.tiff")
        dtype = None
        shifts = False
        flip = True
        clear_images = True
        mkdir = True

        image = "image"
        metadata = "metadata"
        annotated_image = "annotated"

        requested_filepath = Path("path/to/file_Annotated.tiff")

        saver = SaveAnnotatorMixin()

        mocked_create_metadata.return_value = metadata
        saver.get_output = MagicMock(return_value=image)
        saver.referenced = False
        saver.annotated_image = annotated_image

        self.assertTrue(
            saver.save_images(
                filepath,
                dtype,
                shifts,
                flip,
                clear_images,
                mkdir,
                strict=True,
                annotated_path=requested_filepath,
            )
        )

        saver.get_output.assert_called_once_with(
            load=True, shifts=shifts, flip=flip, clear_images=clear_images
        )
        mocked_mkdir.assert_has_calls([call(parents=True, exist_ok=True)] * 2)
        mocked_manual_save.assert_called_once_with(filepath, image, dtype, metadata)
        mocked_ann_save.assert_called_once_with(requested_filepath, annotated_image)

    @patch.object(SaveMixin, "create_metadata")
    @patch("txrm2tiff.txrm.save_mixin.manual_save")
    @patch("pathlib.Path.mkdir")
    def test_save_images_returns_False_without_image(
        self, mocked_mkdir, mocked_manual_save, mocked_create_metadata
    ):
        filepath = Path("path/to/file.ext")
        dtype = None
        shifts = False
        flip = True
        clear_images = True
        mkdir = True

        image = None
        metadata = "metadata"
        annotated_image = "annotated"

        saver = SaveMixin()

        mocked_create_metadata.return_value = metadata
        saver.get_output = MagicMock(return_value=image)
        saver.save_annotations = MagicMock()
        saver.referenced = False
        saver.annotated_image = annotated_image

        self.assertFalse(
            saver.save_images(
                filepath, dtype, shifts, flip, clear_images, mkdir, strict=False
            )
        )

        saver.get_output.assert_called_once_with(
            load=True, shifts=shifts, flip=flip, clear_images=clear_images
        )

        mocked_mkdir.assert_not_called()
        mocked_manual_save.assert_not_called()
        saver.save_annotations.assert_not_called()
