import os
from os import path
import logging
from pathlib import Path


dragndrop_bat_file = Path(__file__).resolve().parent / "scripts" / "dragndrop.bat"

def _create_lnk_file(shortcut_path):
    try:
        # win23com is from the package pywin32, only available in Windows
        import win32com.client
    except ImportError:
        msg = "win32com of pywin32 cannot be imported! Please run 'pip install pywin32' (with '--user' argument if on a shared python environment) then try again."
        print(msg)
        logging.error(msg)
        raise
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(str(shortcut_path))
    shortcut.Targetpath = str(dragndrop_bat_file)
    shortcut.save()


def create_Windows_shortcut():
    if os.name != "nt":
        logging.error("This command is only valid on Windows installations.")
        return 
    else:
        # Place link on users desktop
        shortcut_path = Path.home() / "Desktop" / "txrm2tiff.lnk"
        msg = f"Creating shortcut on user desktop: {shortcut_path}"
        logging.info(msg)
        if path.exists(shortcut_path):
            msg = "txrm2tiff shortcut already found. Are you sure you want to replace it? (y/N)"
            user_input = str(input(msg))
            logging.debug(msg)
            logging.debug("User input: %s", user_input)
            if user_input.lower() == "y" or user_input.lower() == "yes":
                logging.info("The existing shortcut will be replaced.")
            elif user_input.lower() == "n" or user_input.lower() == "no":
                logging.info("The existing shortcut will not be modified.")
            else:
                logging.info("Invalid input: %s. The existing shortcut will not be modified.", user_input)
                return
        _create_lnk_file(shortcut_path)
        print(f"Desktop shortcut created! It can be found here: {shortcut_path}")
        logging.info("Desktop shortcut created! It can be found here: %s", str(shortcut_path))