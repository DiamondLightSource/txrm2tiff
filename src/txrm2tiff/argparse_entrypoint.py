#!/usr/bin/python
from pathlib import Path
import argparse
import sys

from .__init__ import __version__

parser = argparse.ArgumentParser(prog="txrm2tiff", description="Converter of txrm/xrm files to OME tif/tiff files", add_help=False)
convert_group = parser.add_argument_group("Converter arguments")
convert_group.add_argument('-i', '--input', dest='input_path', default=None, type=str, action='store', help='txrm or xrm file to convert, or directory containing txrm and/or xrm files')
convert_group.add_argument('-r', '--reference', dest='custom_reference', default=None, type=str, action='store', help='specify a custom file (tiff, txrm or xrm) for reference (ignores internal reference)')
convert_group.add_argument('-o', '--output-path', dest='output_path', default=None, action='store', help='specify output (can be directory or file) - specified directories that do not exist will be created')
convert_group.add_argument('-a', '--annotate', dest='annotate', action='store_true', help='save a second annotated copy, if annotations are available')
convert_group.add_argument('-d', '--datatype', dest='data_type', default=None, action='store', choices=['uint16', 'float32', 'float64'], help='specify output data type (default: decides data type from input)')
convert_group.add_argument('--ignore-ref', dest='ignore_reference', default=False, action='store_true', help='if specified ignore any internal reference')
convert_group.add_argument('--set-logging', dest='logging_level', default="info", type=str, action='store', help='pick logging level (options: debug, info, warning, error, critical)')

package_group = parser.add_argument_group("Package info")
package_group.add_argument('-v', '--version', action='version', version="%(prog)s {}".format(__version__))
package_group.add_argument('-h', '--help', action='help', help="show this help message and exit")

subparsers = parser.add_subparsers(title="Additional options", help='Additional functionality options')
inspect_parser = subparsers.add_parser(name="inspect", add_help=False)
inspect_info_group = inspect_parser.add_argument_group("Inspection options")
inspect_info_group.add_argument('-i', '--input', dest='input_path', default="", type=str, action='store', help='txrm or xrm file to inspect')
inspect_info_group.add_argument('-e', '--extra', action='store_true', help="Shows additional file info")
inspect_info_group.add_argument('-l', '--list-streams', action='store_true', help="List all streams")
inspect_info_group.add_argument('-s', '--inspect-streams', dest='streams', nargs='+', help="Inspect stream")
inspect_info_group.add_argument('-h', '--help', action='help', help="show this help message and exit")

setup_parser = subparsers.add_parser(name="setup", add_help=False)
setup_info_group = setup_parser.add_argument_group("Setup options")
setup_info_group.add_argument("-w", "--windows-shortcut", action='store_true', help="creates a Windows shortcut on the user's desktop that accepts drag and dropped files (allows batch processing)")
setup_info_group.add_argument('-h', '--help', action='help', help="show this help message and exit")


def main():
    if "inspect" in sys.argv:
        inspect_args = parser.parse_args(namespace=inspect_parser)
        if inspect_args.input_path:
            from .inspector import Inspector
            with Inspector(inspect_args.input_path) as inspector:
                inspector.inspect(extra=bool(inspect_args.extra))
                if inspect_args.list_streams:
                    inspector.list_streams()
                if inspect_args.streams:
                    inspector.inspect_streams(*inspect_args.streams)
                print(inspector.get_text())
                return
        else:
            inspect_parser.print_help()
            return

    if "setup" in sys.argv:
        setup_args = parser.parse_args(namespace=setup_parser)
        if setup_args.windows_shortcut:
            from .shortcut_creator import create_Windows_shortcut
            create_Windows_shortcut()
            return
        else:
            setup_parser.print_help()
            return

    args = parser.parse_args()
    if args.input_path:
        from .run import run
        run(**vars(args))

    else:
        parser.print_help()
        print("\n\n\ntxrm2tiff inspect:\n")
        inspect_parser.print_help()
        print("\n\n\ntxrm2tiff setup:\n")
        setup_parser.print_help()
