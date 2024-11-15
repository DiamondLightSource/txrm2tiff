#!/usr/bin/python
import argparse
import sys
import os

if os.name == "nt":
    try:
        # Forces stream to use utf-8 (only necessary on Windows)
        # https://docs.python.org/3/library/io.html#io.TextIOWrapper.reconfigure
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from .info import __version__
from .utils.metadata import CLI_DTYPES


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="txrm2tiff",
        description="Converter of txrm/xrm files to OME tif/tiff files",
        add_help=False,
    )
    convert_group = parser.add_argument_group("Converter arguments")
    convert_group.add_argument(
        "-i",
        "--input",
        dest="input_path",
        default=None,
        type=str,
        action="store",
        help="txrm or xrm file to convert, or directory containing txrm and/or xrm files",
    )
    convert_group.add_argument(
        "-r",
        "--reference",
        dest="custom_reference",
        default=None,
        type=str,
        action="store",
        help="specify a custom file (tiff, txrm or xrm) for reference (ignores internal reference)",
    )
    convert_group.add_argument(
        "-o",
        "--output-path",
        dest="output_path",
        default=None,
        action="store",
        help="specify output (can be directory or file) - specified directories that do not exist will be created",
    )
    convert_group.add_argument(
        "-a",
        "--annotate",
        dest="annotate",
        action="store_true",
        help="save a second annotated copy, if annotations are available",
    )
    convert_group.add_argument(
        "-f",
        "--flip",
        dest="flip",
        action="store_true",
        help="if specified, the output image will be flipped in the Y-axis with respect to how it displays in XRM Data Explorer",
    )
    convert_group.add_argument(
        "-d",
        "--datatype",
        dest="data_type",
        default=None,
        action="store",
        choices=CLI_DTYPES,
        help="specify output data type (default: decides data type from input)",
    )
    convert_group.add_argument(
        "--apply-shifts",
        dest="apply_shifts",
        default=False,
        action="store_true",
        help="if specified, apply saved shifts (if available)",
    )
    convert_group.add_argument(
        "--ignore-ref",
        dest="ignore_reference",
        default=False,
        action="store_true",
        help="if specified, ignore any internal reference",
    )
    convert_group.add_argument(
        "--set-logging",
        dest="logging_level",
        default="info",
        type=str,
        action="store",
        help="pick logging level (options: debug, info, warning, error, critical)",
    )

    package_group = parser.add_argument_group("Package info")
    package_group.add_argument(
        "-v", "--version", action="version", version="%(prog)s {}".format(__version__)
    )
    package_group.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )

    subparsers = parser.add_subparsers(
        title="Additional options",
        dest="subparser_name",
        help="Additional functionality options",
    )
    inspect_parser = subparsers.add_parser(name="inspect", add_help=False)
    inspect_info_group = inspect_parser.add_argument_group("Inspection options")
    inspect_info_group.add_argument(
        "-i",
        "--input",
        dest="input_path",
        default="",
        type=str,
        required=True,
        action="store",
        help="txrm or xrm file to inspect",
    )
    inspect_info_group.add_argument(
        "-e", "--extra", action="store_true", help="Shows additional file info"
    )
    inspect_info_group.add_argument(
        "-l", "--list-streams", action="store_true", help="List all streams"
    )
    inspect_info_group.add_argument(
        "-s", "--inspect-streams", dest="streams", nargs="+", help="Inspect stream"
    )
    inspect_info_group.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )

    setup_parser = subparsers.add_parser(name="setup", add_help=False)
    setup_info_group = setup_parser.add_argument_group("Setup options")
    setup_info_group.add_argument(
        "-w",
        "--windows-shortcut",
        action="store_true",
        help="creates a Windows shortcut on the user's desktop that accepts drag and dropped files (allows batch processing)",
    )
    setup_info_group.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )
    return parser


def main() -> None:
    parser = create_parser()
    namespace = parser.parse_args()
    if namespace.subparser_name == "inspect":
        from .inspector import Inspector
        from .txrm import open_txrm

        try:
            with open_txrm(
                namespace.input_path,
                load_images=False,
                load_reference=False,
                strict=False,
            ) as txrm:
                inspector = Inspector(txrm)
                inspector.inspect(extra=getattr(namespace, "extra", False))
                if getattr(namespace, "list_streams"):
                    inspector.list_streams()
                if getattr(namespace, "streams"):
                    inspector.inspect_streams(*namespace.streams)
                print(inspector.get_text())
            return
        except Exception as e:
            import traceback

            print(
                "Exception occurred while inspecting %s:\n\n%s"
                % (
                    namespace.input_path,
                    "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                )
            )
        return

    elif namespace.subparser_name == "setup":
        if namespace.windows_shortcut:
            from .utils.shortcut_creation import create_Windows_shortcut

            create_Windows_shortcut()
            return

    args = parser.parse_args()
    if args.input_path:
        from . import convert_and_save

        convert_and_save(**vars(args))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
