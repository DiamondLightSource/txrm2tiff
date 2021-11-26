import unittest
from unittest.mock import MagicMock, patch

import os
import pathlib
from pathlib import Path

from txrm2tiff.utils.shortcut_creation import create_Windows_shortcut, _create_lnk_file


class TestSetupFunctions(unittest.TestCase):
    @patch.object(pathlib.Path, "exists", MagicMock(return_value=False))
    @patch("logging.error")
    def test_setup_windows_shortcut_function_called(self, mocked_error):
        with patch(
            "txrm2tiff.utils.shortcut_creation._create_lnk_file", MagicMock()
        ) as mocked_shortcut_creation:
            create_Windows_shortcut()
            if os.name == "nt":
                mocked_shortcut_creation.assert_called_once_with(
                    Path(os.environ["HOMEDRIVE"])
                    / os.environ["HOMEPATH"]
                    / "Desktop"
                    / "txrm2tiff.lnk"
                )
            else:
                mocked_error.assert_called_once_with(
                    "This command is only valid on Windows installations."
                )

    def test_setup_windows_shortcut_test_created(self):
        # Only run this test if on Windows
        if os.name == "nt":
            test_shortcut_path = Path(".") / "test_shortcut.lnk"
            _create_lnk_file(test_shortcut_path)
            link_created = test_shortcut_path.exists()
            if link_created:
                os.remove(test_shortcut_path)
            self.assertTrue(link_created)
