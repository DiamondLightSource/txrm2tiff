import unittest
from unittest.mock import patch, MagicMock, ANY

import os
import sys
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

from txrm2tiff import __version__
from txrm2tiff import argparse_entrypoint


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
        # Add local intall to path for testing with development install
        if os.name == "posix":
            os.environ["PATH"] += (
                os.pathsep
                + os.path.join(os.path.expanduser("~"), ".local", "bin")
                )
    
    @patch('argparse.ArgumentParser.print_help')
    def test_argparse_function(self, mocked_printhelp):
        sys.argv = ["txrm2tiff"]
        argparse_entrypoint.main()

        mocked_printhelp.assert_called_once()

    @patch('argparse.ArgumentParser.print_help')
    def test_argparse_function_setup_subpaser(self, mocked_printhelp):
        sys.argv = ["txrm2tiff", "setup"]
        argparse_entrypoint.main()

        mocked_printhelp.assert_called_once()

    def test_argparse_function_setup_windows_shortcut(self):
        with patch('txrm2tiff.shortcut_creator.create_Windows_shortcut', MagicMock()) as mocked_shortcut_creator:
            sys.argv = ["txrm2tiff", "setup", "-w"]
            argparse_entrypoint.main()
            
            mocked_shortcut_creator.assert_called_once()

    def test_argparse_function_with_args(self):
        with patch('txrm2tiff.run.run', MagicMock()) as mock_run:
            mock_run.return_value = None
            input_arg = "input_path"
            ref_arg = "ref_path"
            sys.argv =["txrm2tiff", "--input", input_arg, "--reference", ref_arg]
            # argparse_entrypoint imports from the non-
            argparse_entrypoint.main()

            mock_run.assert_called_once_with(input_path=input_arg, custom_reference=ref_arg, output_path=None, ignore_reference=False, logging_level='info')

    def test_script_method(self):
        args = ["path_to/input.txrm"]
        run_args = ["txrm2tiff", "-i", args[0]]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, text=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")
        self.assertIn(f"No such file or directory: {args[0]}", stdout, msg=f"Actual stdout: {stdout}")

    def test_script_method_version_number(self):
        run_args = ["txrm2tiff", "--version"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, text=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")
        self.assertEqual(f"txrm2tiff {__version__}", stdout, msg=f"Actual stdout: {stdout}")

    def test_script_method_setup_subparser(self):
        run_args = ["txrm2tiff", "setup"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, text=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")
        self.assertIn(f"txrm2tiff setup [-w] [-h]", stdout, msg=f"Actual stdout: {stdout}")

    
    def test_module_without_arguments_returns_help(self):
        run_args = [sys.executable, "-m", "txrm2tiff"]
        path = os.environ["PATH"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, text=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")
        self.assertIn("Converter of txrm/xrm files to OME tif/tiff files", stdout, msg=f"Actual stdout: {stdout}")
        
    def test_module_function_setup_windows_shortcut(self):
        run_args = [sys.executable, "-m", "txrm2tiff", "setup", "-w"]

        if os.name != "nt":
            expected_stdout = "This command is only valid on Windows installations."
        else:
            expected_stdout = "Desktop shortcut created! It can be found here: "
            desktop_path = Path.home() / "Desktop" / "txrm2tiff.lnk"
            if desktop_path.exists():
                desktop_path.unlink()

        with Popen(run_args, stdout=PIPE, stderr=STDOUT, text=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")

        self.assertIn(expected_stdout, stdout, msg=f"Actual stdout: {stdout}")


    def test_module_function_with_args(self):
        input_arg = "input_file_path"
        ref_arg = "ref_path"
        run_args = [sys.executable, "-m", "txrm2tiff", "--input", input_arg, "--reference", ref_arg]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, text=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")

        self.assertIn(f"No such file or directory: {input_arg}", stdout, msg=f"Actual stdout: {stdout}")

    def test_module_method(self):
        args = ["path_to/input.txrm"]
        run_args = [sys.executable, "-m", "txrm2tiff", "-i", args[0]]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, text=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")
        self.assertIn(f"No such file or directory: {args[0]}", stdout, msg=f"Actual stdout: {stdout}")

    def test_module_method_version_number(self):
        run_args = [sys.executable, "-m", "txrm2tiff", "--version"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, text=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")
        self.assertEqual(f"txrm2tiff {__version__}", stdout, msg=f"Actual stdout: {stdout}")

    def test_module_method_setup_subparser_without_arguments_returns_help(self):
        run_args = [sys.executable, "-m", "txrm2tiff", "setup"]
        with Popen(run_args, stdout=PIPE, stderr=STDOUT, text=True) as p:
            stdout, _ = p.communicate()
        stdout = stdout.strip("\r\n").strip("\n")
        self.assertIn(f"txrm2tiff setup [-w] [-h]", stdout, msg=f"Actual stdout: {stdout}")
