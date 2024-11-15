import unittest
from unittest.mock import patch, MagicMock, call
from parameterized import parameterized

from pathlib import Path
from tempfile import TemporaryDirectory
from random import choice, randint
from string import ascii_letters

from txrm2tiff.main import (
    convert_and_save,
    _batch_convert_files,
    _convert_and_save,
    _set_output_suffix,
    _convert_file,
    _decide_output_path,
    _find_files,
)


from txrm2tiff.txrm.v3 import Txrm3
from txrm2tiff.txrm.v5 import Txrm5


def generate_random_strings(return_num: int):
    for i in range(return_num):
        yield f"{''.join([choice(ascii_letters) for _ in range(10)])}"


class TestRun(unittest.TestCase):
    def test_set_output_suffix(self):
        txrm_output = _set_output_suffix(Path("file.txrm"))
        self.assertEqual("file.ome.tiff", str(txrm_output))
        xrm_output = _set_output_suffix(Path("file.xrm"))
        self.assertEqual("file.ome.tiff", str(xrm_output))
        txrm_output2 = _set_output_suffix(Path("file.extension"), ".txrm")
        self.assertEqual("file.ome.tiff", str(txrm_output2))
        xrm_output2 = _set_output_suffix(Path("file.extension"), ".xrm")
        self.assertEqual("file.ome.tiff", str(xrm_output2))
        with self.assertRaises(NameError):
            _set_output_suffix(Path("file.bad_extension"), None)

    @parameterized.expand([(Txrm3,), (Txrm5,)])
    @patch("txrm2tiff.main.open_txrm")
    def test_convert_and_save(self, TxrmClass, mocked_open_txrm):
        input_filepath = Path("test_file.txrm")

        txrm = MagicMock(auto_spec=TxrmClass)
        txrm.path = input_filepath
        txrm.name = input_filepath.name
        mocked_open_txrm.return_value.__enter__.return_value = txrm
        _convert_and_save(input_filepath, None, None, False, False, None, False)
        mocked_open_txrm.assert_called_with(input_filepath)
        txrm.save_images.assert_called_with(
            input_filepath.with_suffix(".ome.tiff"),
            None,
            flip=False,
            shifts=False,
            mkdir=True,
        )

    @parameterized.expand([(Txrm3,), (Txrm5,)])
    def test_convert_file(self, TxrmClass):
        input_filepath = Path("test_file.txrm")
        txrm = MagicMock(auto_spec=TxrmClass)
        txrm.path = input_filepath
        txrm.name = input_filepath.name
        _convert_file(txrm, None, False, False)
        txrm.apply_reference.assert_called_once_with(
            None
        )  # Called with None applies internal reference
        txrm.annotate.assert_not_called()

    @parameterized.expand([(Txrm3,), (Txrm5,)])
    def test_convert_file_ignore_reference(self, TxrmClass):
        input_filepath = Path("test_file.txrm")
        txrm = MagicMock(auto_spec=TxrmClass)
        txrm.path = input_filepath
        txrm.name = input_filepath.name
        _convert_file(txrm, None, True, False)
        txrm.apply_reference.assert_not_called()
        txrm.annotate.assert_not_called()

    @parameterized.expand([(Txrm3,), (Txrm5,)])
    def test_convert_file_custom_reference_overrules_ignore_reference(self, TxrmClass):
        input_filepath = Path("test_file.txrm")
        custom_ref = Path("path/custom_ref")
        txrm = MagicMock(auto_spec=TxrmClass)
        txrm.name = input_filepath.name
        _convert_file(txrm, custom_ref, True, False)
        txrm.apply_reference.assert_called_once_with(custom_ref)
        txrm.annotate.assert_not_called()

    @parameterized.expand([(Txrm3,), (Txrm5,)])
    def test_convert_file_with_annotation(self, TxrmClass):
        input_filepath = Path("test_file.txrm")
        txrm = MagicMock(auto_spec=TxrmClass)
        txrm.path = input_filepath
        txrm.name = input_filepath.name
        _convert_file(txrm, None, True, True)
        txrm.annotate.assert_called_once_with()

    @parameterized.expand([(".txrm", ".ome.tiff"), (".xrm", ".ome.tiff")])
    def test_decide_output_path(self, input_suffix, output_suffix):
        filepath = Path("./path/to/thisisafile.ext")
        output = _decide_output_path(filepath.with_suffix(input_suffix), None)
        self.assertEqual(output, filepath.with_suffix(output_suffix))

    @parameterized.expand([(".txrm", ".ome.tiff"), (".xrm", ".ome.tiff")])
    def test_decide_output_path_with_output_dir(self, input_suffix, output_suffix):
        filepath = Path("./path/to/thisisafile.ext")
        output_dir = Path("./a/different/path")
        output = _decide_output_path(filepath.with_suffix(input_suffix), output_dir)
        self.assertEqual(
            output, (output_dir / filepath.name).with_suffix(output_suffix)
        )

    @parameterized.expand([(".txrm"), (".xrm")])
    def test_decide_output_path_with_output_path(self, input_suffix):
        filepath = Path("./path/to/thisisafile.ext")
        output_path = Path("./a/different/path.ext")
        output = _decide_output_path(filepath.with_suffix(input_suffix), output_path)
        self.assertEqual(output, output_path)

    def test_find_files(self):
        with TemporaryDirectory(prefix="test_finds_files_", dir=".") as tmpdir:
            test_dir = Path(tmpdir)
            txrm_subdir = test_dir / "txrm_subdir"
            txrm_subdir.mkdir()
            xrm_subdir = test_dir / "xrm_subdir"
            xrm_subdir.mkdir()
            bad_subdir = test_dir / "bad_subdir"
            bad_subdir.mkdir()
            num_txrm = 10
            num_xrm = 5
            num_bad = 7
            txrm_files = {
                txrm_subdir / f"{fname}.txrm"
                for fname in generate_random_strings(num_txrm)
            }

            xrm_files = {
                xrm_subdir / f"{fname}.xxrm"
                for fname in generate_random_strings(num_xrm)
            }
            bad_files = {
                bad_subdir / f"{fname}.ext"
                for fname in generate_random_strings(num_bad)
            }
            for f in set().union(txrm_files, xrm_files, bad_files):
                f.touch()

            self.assertEqual(txrm_files.union(xrm_files), set(_find_files(test_dir)))

    @patch("txrm2tiff.main._set_output_suffix")
    @patch("txrm2tiff.main._find_files")
    @patch("txrm2tiff.main._convert_and_save")
    def test_batch_convert_files(
        self, mocked_convert_save, mocked_find_files, mocked_suffix_definer
    ):
        input_dir = Path("./path/to/thisisadir")
        num_txrm = 2
        num_xrm = 3
        txrm_files = [
            input_dir / "txrm_dir" / f"{fname}.txrm"
            for fname in generate_random_strings(num_txrm)
        ]

        xrm_files = [
            input_dir / f"{fname}.xrm" for fname in generate_random_strings(num_xrm)
        ]
        input_files = txrm_files + xrm_files
        output_files = [f.with_suffix(".ome.tiff") for f in txrm_files] + [
            f.with_suffix(".ome.tiff") for f in xrm_files
        ]
        mocked_find_files.return_value = input_files
        mocked_suffix_definer.side_effect = output_files
        _batch_convert_files(input_dir)
        mocked_convert_save.assert_has_calls(
            [
                call(f_in, f_out, None, True, False, None, False, False)
                for f_in, f_out in zip(input_files, output_files)
            ]
        )

    @patch("pathlib.Path.mkdir", MagicMock())
    @patch("txrm2tiff.main._find_files")
    @patch("txrm2tiff.main._convert_and_save")
    def test_batch_convert_files_to_output(
        self,
        mocked_convert_save,
        mocked_find_files,
    ):
        input_dir = Path("./path/to/thisisadir")
        output_dir = Path("./path/to/output_dir")
        num_txrm = 3
        num_xrm = 2
        txrm_files = [
            Path(f"txrm_dir/{fname}.txrm")
            for fname in generate_random_strings(num_txrm)
        ]

        xrm_files = [Path(f"{fname}.xrm") for fname in generate_random_strings(num_xrm)]
        input_files = txrm_files + xrm_files
        output_files = [f.with_suffix(".ome.tiff") for f in txrm_files] + [
            f.with_suffix(".ome.tiff") for f in xrm_files
        ]
        input_paths = [input_dir / s for s in input_files]
        output_paths = [output_dir / s for s in output_files]
        mocked_find_files.return_value = input_paths
        _batch_convert_files(input_dir, output_dir)
        mocked_convert_save.assert_has_calls(
            [
                call(f_in, f_out, None, True, False, None, False, False)
                for f_in, f_out in zip(input_paths, output_paths)
            ]
        )

    @patch("pathlib.Path.exists", MagicMock(return_value=False))
    def test_convert_and_save_non_existant_file(self):
        input_path = "input/path/file.txrm"
        custom_reference = "input/path/custom.xrm"
        output_path = "output/path"
        dtype = "uint8"
        logging_level = "debug"
        with self.assertRaises(IOError):
            convert_and_save(
                input_path,
                output_path,
                custom_reference,
                True,
                False,
                dtype,
                False,
                logging_level,
            )

    @patch("pathlib.Path.exists", MagicMock(return_value=True))
    @patch("txrm2tiff.main.create_logger")
    @patch("txrm2tiff.main._convert_and_save")
    def test_convert_and_save_str_input_invalid_custom_reference(
        self, mocked_convert_and_save, mocked_create_logger
    ):
        input_path = "input/path/file.txrm"
        custom_reference = "input/path/custom.xrm"
        output_path = "output/path"
        dtype = "uint8"
        logging_level = "debug"

        convert_and_save(
            input_path,
            output_path,
            custom_reference,
            True,
            False,
            dtype,
            False,
            False,
            logging_level,
        )

        mocked_create_logger.assert_called_once_with(logging_level)
        mocked_convert_and_save.assert_called_once_with(
            Path(input_path), Path(output_path), None, True, False, dtype, False, False
        )  # Casts paths to Path

    @patch("pathlib.Path.exists", MagicMock(return_value=True))
    @patch("txrm2tiff.main.create_logger")
    @patch("txrm2tiff.main._convert_and_save")
    def test_convert_and_save_Path_input_invalid_custom_reference(
        self, mocked_convert_and_save, mocked_create_logger
    ):
        input_path = Path("input/path/file.txrm")
        custom_reference = Path("input/path/custom.xrm")
        output_path = Path("output/path")
        dtype = "uint8"
        logging_level = "debug"

        convert_and_save(
            input_path,
            output_path,
            custom_reference,
            True,
            False,
            dtype,
            True,
            False,
            logging_level,
        )

        mocked_create_logger.assert_called_once_with(logging_level)
        mocked_convert_and_save.assert_called_once_with(
            input_path, output_path, None, True, False, dtype, True, False
        )

    @patch("pathlib.Path.is_file", MagicMock(return_value=True))
    @patch("pathlib.Path.exists", MagicMock(return_value=True))
    @patch("txrm2tiff.main.create_logger")
    @patch("txrm2tiff.main._convert_and_save")
    def test_convert_and_save_Path_input_custom_reference(
        self, mocked_convert_and_save, mocked_create_logger
    ):
        input_path = Path("input/path/file.txrm")
        custom_reference = Path("input/path/custom.xrm")
        output_path = Path("output/path")
        dtype = "uint8"
        logging_level = "debug"

        convert_and_save(
            input_path,
            output_path,
            custom_reference,
            True,
            False,
            dtype,
            True,
            False,
            logging_level,
        )

        mocked_create_logger.assert_called_once_with(logging_level)
        mocked_convert_and_save.assert_called_once_with(
            input_path, output_path, custom_reference, True, False, dtype, True, False
        )

    @patch("pathlib.Path.is_dir", MagicMock(return_value=True))
    @patch("pathlib.Path.exists", MagicMock(return_value=True))
    @patch("txrm2tiff.main.create_logger")
    @patch("txrm2tiff.main._batch_convert_files")
    def test_convert_and_save_str_input_batch(
        self, mocked_batch_convert_files, mocked_create_logger
    ):
        input_path = "input/path"
        output_path = "output/path"
        dtype = "uint8"
        logging_level = "debug"

        convert_and_save(
            input_path, output_path, None, True, False, dtype, True, True, logging_level
        )

        mocked_create_logger.assert_called_once_with(logging_level)
        mocked_batch_convert_files.assert_called_once_with(
            Path(input_path), Path(output_path), True, False, dtype, True, True
        )

    @patch("pathlib.Path.is_dir", MagicMock(return_value=True))
    @patch("pathlib.Path.exists", MagicMock(return_value=True))
    @patch("txrm2tiff.main.create_logger")
    @patch("txrm2tiff.main._batch_convert_files")
    def test_convert_and_save_Path_input_batch(
        self, mocked_batch_convert_files, mocked_create_logger
    ):
        input_path = Path("input/path")
        output_path = Path("output/path")
        dtype = "uint8"
        logging_level = "debug"

        convert_and_save(
            input_path,
            output_path,
            None,
            True,
            False,
            dtype,
            True,
            False,
            logging_level,
        )

        mocked_create_logger.assert_called_once_with(logging_level)
        mocked_batch_convert_files.assert_called_once_with(
            input_path, output_path, True, False, dtype, True, False
        )

    @patch("txrm2tiff.main._convert_and_save")
    def test_convert_and_save_with_file(self, mocked_convert):
        with TemporaryDirectory(dir=".") as tmp_in:
            tmp_in_filepath = Path(tmp_in) / f"{randint(0,9999)}.xrm"
            tmp_in_filepath.touch()
            convert_and_save(tmp_in_filepath)
        mocked_convert.assert_called_with(
            tmp_in_filepath, None, None, False, False, None, False, False
        )

    @patch("txrm2tiff.main._batch_convert_files")
    def test_convert_and_save_with_dir(self, mocked_batch_convert):
        with TemporaryDirectory(dir=".") as tmp_in:
            tmp_in_path = Path(tmp_in)
            convert_and_save(tmp_in_path)
        mocked_batch_convert.assert_called_with(
            tmp_in_path, None, False, False, None, False, False
        )
