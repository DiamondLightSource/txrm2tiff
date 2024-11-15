import unittest
from unittest.mock import patch, MagicMock

import os
import sys
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

from txrm2tiff import __version__
from txrm2tiff import __main__


class TestCommandlineEntryPoint(unittest.TestCase):
    # Before these running tests, the module must be installed in development mode using:
    # `pip install --editable . --user` (from the project folder)
    # it can be uninstalled at the end using:
    # `pip uninstall txrm2tiff`
    # IF TESTING ON WINDOWS: make sure the correct python installation is set as the default in PATH

    @classmethod
    def setUpClass(cls):
        # Set maximum difference string length to None (infinite)
        cls.maxDiff = None
        # Add local install to path for testing with development install
        if os.name == "posix":
            os.environ["PATH"] += os.pathsep + os.path.join(
                os.path.expanduser("~"), ".local", "bin"
            )

    @patch("argparse.ArgumentParser.print_help")
    def test_argparse_function(self, mocked_printhelp):
        sys.argv = ["txrm2tiff"]
        __main__.main()

        mocked_printhelp.assert_called()

    @patch("argparse.ArgumentParser.print_help")
    def test_argparse_function_setup_subpaser(self, mocked_printhelp):
        sys.argv = ["txrm2tiff", "setup"]
        __main__.main()

        mocked_printhelp.assert_called_once_with()

    def test_argparse_function_setup_windows_shortcut(self):
        with patch(
            "txrm2tiff.utils.shortcut_creation.create_Windows_shortcut", MagicMock()
        ) as mocked_shortcut_creation:
            sys.argv = ["txrm2tiff", "setup", "-w"]
            __main__.main()

            mocked_shortcut_creation.assert_called_once_with()

    def test_argparse_function_with_args(self):
        with patch("txrm2tiff.convert_and_save", MagicMock()) as mock_convert_and_save:
            mock_convert_and_save.return_value = None
            input_arg = "input_path"
            annotate_arg = False
            flip = False
            ref_arg = None
            data_type_arg = None
            sys.argv = ["txrm2tiff", "--input", input_arg]
            __main__.main()

            mock_convert_and_save.assert_called_once_with(
                input_path=input_arg,
                custom_reference=ref_arg,
                output_path=None,
                annotate=annotate_arg,
                flip=flip,
                data_type=data_type_arg,
                apply_shifts=False,
                ignore_reference=False,
                logging_level="info",
            )

    def test_argparse_function_with_all_args(self):
        with patch("txrm2tiff.convert_and_save", MagicMock()) as mock_convert_and_save:
            mock_convert_and_save.return_value = None
            input_arg = "input_path"
            annotate_arg = True
            flip_arg = True
            ref_arg = "ref_path"
            data_type_arg = "uint16"
            sys.argv = [
                "txrm2tiff",
                "--input",
                input_arg,
                "--reference",
                ref_arg,
                "--datatype",
                data_type_arg,
                "--annotate",
                "--flip",
                "--apply-shifts",
                "--ignore-ref",
            ]
            __main__.main()

            mock_convert_and_save.assert_called_once_with(
                input_path=input_arg,
                custom_reference=ref_arg,
                output_path=None,
                annotate=annotate_arg,
                flip=flip_arg,
                data_type=data_type_arg,
                apply_shifts=True,
                ignore_reference=True,
                logging_level="info",
            )

    def test_script_method(self):
        args = [Path("path_to/input.txrm")]
        run_args = ["txrm2tiff", "-i", str(args[0])]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, universal_newlines=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")
        self.assertIn(
            f"No such file or directory: {args[0]}",
            stdout,
            msg=f"Actual stdout: {stdout}",
        )

    def test_script_method_version_number(self):
        run_args = ["txrm2tiff", "--version"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, universal_newlines=True) as p:
            stdout, _ = p.communicate()
        # Get the last line (ignore the input)
        stdout = [_.strip() for _ in stdout.strip("\n").split("\n")][-1]
        self.assertEqual(
            f"txrm2tiff {__version__}", stdout, msg=f"Actual stdout: {stdout}"
        )

    def test_script_method_setup_subparser(self):
        run_args = ["txrm2tiff", "setup"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, universal_newlines=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.replace("\r\n", " ").replace("\n", " ")
        self.assertIn(
            "usage: txrm2tiff setup [-w] [-h]", stdout, msg=f"Actual stdout: {stdout}"
        )

    def test_script_method_inspect_subparser(self):
        run_args = ["txrm2tiff", "inspect"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, universal_newlines=True) as p:
            stdout, _ = p.communicate()
        stdout = " ".join(
            [text.strip() for text in stdout.replace("\r\n", " ").split("\n")]
        )
        self.assertIn(
            "txrm2tiff inspect [-i INPUT_PATH] [-e] [-l] [-s STREAMS [STREAMS ...]] [-h]",
            stdout,
            msg=f"Actual stdout: {stdout}",
        )

    def test_module_without_arguments_returns_help(self):
        run_args = [sys.executable, "-m", "txrm2tiff"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, universal_newlines=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")
        self.assertIn(
            "Converter of txrm/xrm files to OME tif/tiff files",
            stdout,
            msg=f"Actual stdout: {stdout}",
        )

    def test_module_function_with_args(self):
        input_path = "input_file_path"
        custom_reference = "ref_path"
        output_path = None
        annotate = True
        flip = True
        data_type = None
        apply_shifts = False
        ignore_reference = False
        logging_level = 1

        run_args = [
            sys.executable,
            "-m",
            "txrm2tiff",
            "--input",
            input_path,
            "--reference",
            custom_reference,
            "--set-logging",
            str(logging_level),
            "--annotate",
            "--flip",
        ]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, universal_newlines=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.replace("\r\n", " ").replace("\n", " ")
        self.assertIn(
            f"Running with arguments: input_path={input_path}, custom_reference={custom_reference}, output_path={output_path}, annotate={annotate}, flip={flip}, data_type={data_type}, apply_shifts={apply_shifts}, ignore_reference={ignore_reference}, logging_level={logging_level}",
            stdout,
            msg=f"Actual stdout: {stdout}",
        )
        self.assertIn(
            f"No such file or directory: {input_path}",
            stdout,
            msg=f"Actual stdout: {stdout}",
        )

    def test_module_method(self):
        input_path = Path("path_to/input.txrm")
        custom_reference = None
        output_path = None
        annotate = False
        flip = False
        data_type = None
        apply_shifts = False
        ignore_reference = False
        logging_level = 1
        args = [input_path]
        run_args = [
            sys.executable,
            "-m",
            "txrm2tiff",
            "-i",
            str(args[0]),
            "--set-logging",
            str(logging_level),
        ]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, universal_newlines=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.replace("\r\n", " ").replace("\n", " ")
        self.assertIn(
            f"Running with arguments: input_path={input_path}, custom_reference={custom_reference}, output_path={output_path}, annotate={annotate}, flip={flip}, data_type={data_type}, apply_shifts={apply_shifts}, ignore_reference={ignore_reference}, logging_level={logging_level}",
            stdout,
            msg=f"Actual stdout: {stdout}",
        )
        self.assertIn(
            f"No such file or directory: {input_path}",
            stdout,
            msg=f"Actual stdout: {stdout}",
        )

    def test_module_method_version_number(self):
        run_args = [sys.executable, "-m", "txrm2tiff", "--version"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, universal_newlines=True) as p:
            stdout, _ = p.communicate()
        # Get the last line (ignore the input)
        stdout = [_.strip() for _ in stdout.strip("\n").split("\n")][-1]
        self.assertEqual(
            f"txrm2tiff {__version__}", stdout, msg=f"Actual stdout: {stdout}"
        )

    def test_module_method_setup_subparser_without_arguments_returns_help(self):
        run_args = [sys.executable, "-m", "txrm2tiff", "setup"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, universal_newlines=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")
        self.assertIn(
            "txrm2tiff setup [-w] [-h]", stdout, msg=f"Actual stdout: {stdout}"
        )
