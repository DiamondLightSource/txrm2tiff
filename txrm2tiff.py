#!/usr/bin/python
from pathlib import Path
import sys
from src.run import run


def main(argv):
    run(*argv)


if __name__ == "__main__":
    main(sys.argv[1:])
