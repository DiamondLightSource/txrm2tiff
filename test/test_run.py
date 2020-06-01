import unittest
from unittest.mock import patch, MagicMock, call

from pathlib import Path
from tempfile import TemporaryDirectory
from random import randint, sample

from src.txrm2tiff.run import run, _batch_convert_files, _convert_and_save, _define_output_suffix, TxrmToImage

class TestRun(unittest.TestCase):

    def test_define_output_suffix(self):
        txrm_output = _define_output_suffix(Path("file.txrm"))
        self.assertEqual("file.ome.tiff", str(txrm_output))
        xrm_output = _define_output_suffix(Path("file.xrm"))
        self.assertEqual("file.ome.tif", str(xrm_output))
        txrm_output2 = _define_output_suffix(Path("file.extension"), ".txrm")
        self.assertEqual("file.ome.tiff", str(txrm_output2))
        xrm_output2 = _define_output_suffix(Path("file.extension"), ".xrm")
        self.assertEqual("file.ome.tif", str(xrm_output2))
        with self.assertRaises(NameError):
            _define_output_suffix(Path("file.bad_extension"), None)

    def test_define_output_suffix(self):
        txrm_output = _define_output_suffix(Path("file.txrm"))
        xrm_output = _define_output_suffix(Path("file.xrm"))
        self.assertEqual("file.ome.tiff", str(txrm_output))
        self.assertEqual("file.ome.tif", str(xrm_output))

    @patch('src.txrm2tiff.run.file_can_be_opened', MagicMock(return_value=True))
    @patch('src.txrm2tiff.run.ole_file_works', MagicMock(return_value=True))
    @patch.object(TxrmToImage, 'convert')
    @patch.object(TxrmToImage, 'save')
    def test_convert_and_save(self, mocked_save, mocked_convert):
        input_filepath = Path("test_file.txrm")

        _convert_and_save(input_filepath, None, None, False)

        mocked_convert.assert_called_with(input_filepath, None, False)
        mocked_save.assert_called_with(_define_output_suffix(input_filepath))

    @patch('pathlib.Path.mkdir', MagicMock())
    @patch('src.txrm2tiff.run.file_can_be_opened', MagicMock(return_value=True))
    @patch('src.txrm2tiff.run.ole_file_works', MagicMock(return_value=True))
    @patch.object(TxrmToImage, 'convert')
    @patch.object(TxrmToImage, 'save')
    def test_convert_and_save_with_str_output(self, mocked_save, mocked_convert):
        input_filepath = Path("test_file.txrm")
        output_str = "./output/file.extension"
        _convert_and_save(input_filepath, output_str, None, False)

        mocked_convert.assert_called_with(input_filepath, None, False)
        mocked_save.assert_called_with(Path(output_str).with_suffix('.ome.tiff'))

    @patch('pathlib.Path.mkdir', MagicMock())
    @patch('src.txrm2tiff.run.file_can_be_opened', MagicMock(return_value=True))
    @patch('src.txrm2tiff.run.ole_file_works', MagicMock(return_value=True))
    @patch.object(TxrmToImage, 'convert')
    @patch.object(TxrmToImage, 'save')
    def test_convert_and_save_with_dir_Path_output(self, mocked_save, mocked_convert):
        input_filepath = Path("test_file.txrm")
        output_filepath = Path("./output/")

        _convert_and_save(input_filepath, output_filepath, None, False)

        mocked_convert.assert_called_with(input_filepath, None, False)
        mocked_save.assert_called_with(_define_output_suffix(output_filepath / input_filepath.name))

    @patch('src.txrm2tiff.run.file_can_be_opened', MagicMock(return_value=True))
    @patch('src.txrm2tiff.run.ole_file_works', MagicMock(return_value=True))
    @patch.object(TxrmToImage, 'convert')
    @patch.object(TxrmToImage, 'save')
    def test_convert_and_save_with_invalid_output(self, mocked_save, mocked_convert):
        input_filepath = Path("test_file.txrm")

        _convert_and_save(input_filepath, 12345, None, False)

        mocked_convert.assert_called_with(input_filepath, None, False)
        mocked_save.assert_called_with(_define_output_suffix(input_filepath))

    @patch('src.txrm2tiff.run._convert_and_save')
    def test_batch_convert_files_basic(self, mocked_convert):
        with TemporaryDirectory(dir=".") as tmpdir:
            tmppath = Path(tmpdir)
            num_files = randint(5, 10)
            fake_file_list = []
            for i in sample(range(0, 99999), num_files):
                fake_file = (tmppath / f"{i}.txrm")
                fake_file.touch()
                fake_file_list.append(fake_file)
            _batch_convert_files(tmppath, None, False)
        call_list = []
        for fake_file in fake_file_list:
            output_path = _define_output_suffix(fake_file)
            call_list.append(call(fake_file, output_path, None, False))
        mocked_convert.assert_has_calls(call_list, any_order=True)

    @patch('src.txrm2tiff.run._convert_and_save')
    def test_batch_convert_files_with_output_and_deep_dir(self, mocked_convert):
        with TemporaryDirectory(dir=".") as tmp_in:
            with TemporaryDirectory(dir=".") as tmp_out:
                tmp_in_path = Path(tmp_in)
                tmp_out_path = Path(tmp_out)
                tmp_in_path_deep = tmp_in_path / "deep" / "dirs"
                tmp_out_deep = tmp_out_path / "deep" / "dirs"
                tmp_in_path_deep.mkdir(parents=True)
                num_files = randint(5, 10)
                fake_file_list = []
                for i in sample(range(0, 99999), num_files):
                    fake_file = (tmp_in_path_deep / f"{i}.txrm")
                    fake_file.touch()
                    fake_file_list.append(fake_file)
                _batch_convert_files(tmp_in_path, tmp_out, False)
                self.assertTrue(tmp_out_deep.exists())
        call_list = []
        for fake_file in fake_file_list:
            output_path = _define_output_suffix(fake_file)
            call_list.append(call(fake_file, tmp_out_deep / output_path.name, None, False))
        mocked_convert.assert_has_calls(call_list, any_order=True)

    @patch('src.txrm2tiff.run._convert_and_save')
    def test_run_with_file(self, mocked_convert):
        with TemporaryDirectory(dir=".") as tmp_in:
            tmp_in_filepath = Path(tmp_in) / f"{randint(0,9999)}.xrm"
            tmp_in_filepath.touch()
            run(tmp_in_filepath)
        mocked_convert.assert_called_with(tmp_in_filepath, None, None, False)
   
    @patch('src.txrm2tiff.run._batch_convert_files')
    def test_run_with_dir(self, mocked_batch_convert):
        with TemporaryDirectory(dir=".") as tmp_in:
            tmp_in_path = Path(tmp_in)
            run(tmp_in_path)
        mocked_batch_convert.assert_called_with(tmp_in_path, None, False)
