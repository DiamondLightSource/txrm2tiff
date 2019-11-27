#!/usr/bin/python
from pathlib import Path
import sys
sys.path.append("src")
from run import run
import argparse

p = argparse.ArgumentParser()
p.add_argument('input', action='store')
p.add_argument('--reference-using', dest='custom_reference', action='store', default=None)
p.add_argument('--output', dest='output', action='store', default=None)
p.add_argument('--ignore-ref', dest='ignore_reference', action='store_true', default=False)

if __name__ == "__main__":
    args = p.parse_args()
    run(**vars(args))
