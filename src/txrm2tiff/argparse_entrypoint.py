#!/usr/bin/python
from pathlib import Path
import argparse
import sys

from .__init__ import __version__

parser = argparse.ArgumentParser(prog="txrm2tiff", description="Converter of txrm/xrm files to OME tif/tiff files", add_help=False)
convert_group = parser.add_argument_group("Converter arguments")
convert_group.add_argument('-i', '--input', dest='input_path', default="", type=str, action='store', help='txrm or xrm file to convert, or directory containing txrm and/or xrm files')
convert_group.add_argument('-r', '--reference', dest='custom_reference', default="", type=str, action='store', help='specify a custom file (tiff, txrm or xrm) for reference (ignores internal reference)')
convert_group.add_argument('-o', '--output-path', dest='output_path', default=None, action='store', help='specify output (can be directory or file) - specified directories that do not exist will be created')
convert_group.add_argument('--ignore-ref', dest='ignore_reference', default=False, action='store_true', help='if specified ignore any internal reference')
convert_group.add_argument('--set-logging', dest='logging_level', default="info", type=str, action='store', help='pick logging level (options: debug, info, warning, error, critical)')

package_group = parser.add_argument_group("Package info")
package_group.add_argument('-v', '--version', action='version', version="%(prog)s {}".format(__version__))
package_group.add_argument('-h', '--help', action='help', help="show this help message and exit")

setup_subparsers = parser.add_subparsers(title="Setup options", description="enter `txrm2tiff setup -h` for details")
setup_parser = setup_subparsers.add_parser(name="setup", add_help=False)
setup_parser.add_argument("-w", "--windows-shortcut", action='store_true', help="creates a Windows shortcut on the user's desktop that accepts drag and dropped files (allows batch processing)")
setup_info_group = setup_parser.add_argument_group("Setup info")
setup_info_group.add_argument('-h', '--help', action='help', help="show this help message and exit")

def main():
    args = parser.parse_args()
    if hasattr(args, 'windows_shortcut'):
        if args.windows_shortcut:
            from .shortcut_creator import create_Windows_shortcut
            create_Windows_shortcut()
        else:
            setup_parser.print_help()
        return
    if len(sys.argv) > 1 and args.input_path:
        from .run import run
        if not args.custom_reference:
            args.custom_reference = None
        run(**vars(args))

    else:
        parser.print_help()
