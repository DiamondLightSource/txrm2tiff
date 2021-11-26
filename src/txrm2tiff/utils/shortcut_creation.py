import os
import logging
from pathlib import Path
from sys import executable

def _get_Windows_home_path():
    home = os.getenv("USERPROFILE")
    if home is None:
        homedrive = os.getenv("HOMEDRIVE")
        homepath = os.getenv("HOMEPATH")
        if homedrive and homepath:
            home = os.path.join(homedrive, homepath)
        else:
            home = os.getenv("HOME")

    if home is not None and "Users" in home:
        try:
            home_path = Path(home).resolve(strict=True)
            return home_path
        except FileNotFoundError:
            pass
    logging.error("Cannot find valid home path. The following path was found: '%s'", home)
    raise FileNotFoundError(f"Cannot find valid home path. The following path was found: '{home}''")


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
    shortcut.Targetpath = f'"{executable}"'
    shortcut.Arguments = "-m stitch_m"
    shortcut.save()
    msg = f"Shortcut created! It can be found here: {shortcut_path}"
    print(msg)
    logging.info(msg)


def create_Windows_shortcut():
    if os.name != "nt":
        logging.error("This command is only valid on Windows installations.")
        return
    # Place link on users desktop
    shortcut_path = _get_Windows_home_path() / "Desktop" / "txrm2tiff.lnk"
    msg = f"Creating shortcut on user desktop: {shortcut_path}"
    logging.info(msg)
    if shortcut_path.exists():
        msg = f"Existing txrm2tiff shortcut found:'{shortcut_path}'. Are you sure you want to replace it? (y/N)"
        user_input = str(input(msg))
        logging.debug(msg)
        logging.debug("User input: %s", user_input)
        if user_input.lower() == "y" or user_input.lower() == "yes":
            logging.info("The existing shortcut will be replaced.")
        elif user_input.lower() == "n" or user_input.lower() == "no":
            logging.info("The existing shortcut will not be modified.")
            return
        else:
            logging.info("Invalid input: %s. The existing shortcut will not be modified.", user_input)
            return
    _create_lnk_file(shortcut_path)
