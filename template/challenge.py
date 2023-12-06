#!/usr/bin/env python

import argparse
import os
import re
import sys

from dataclasses import dataclass
from pathlib import Path

DEBUG = os.environ.get('DEBUG') is not None

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs, file=sys.stderr)

def evaluate(data: list[str]):
    part_1_result: list[int] = [ line for line in data ]
    part_2_result: list[int] = [ line for line in data ]

    print(f"Part 1: result: {sum(part_1_result)}")
    print(f"Part 2: result: {sum(part_2_result)}")

def parse_args() -> dict[str,str]:
    def is_file(path):
        path = Path(path)
        if path.is_file() and path.exists():
            return Path(path).resolve()
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a file, or doesn't exist")

    formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=100)
    parser = argparse.ArgumentParser(formatter_class = formatter_class)

    default_file = Path(__file__).parent.joinpath('input.txt')
    parser.add_argument('--input-data', '-i', metavar="FILE", dest="file", help='path to input file', type=is_file, required=False, default=default_file)

    args, _ = parser.parse_known_args()
    
    return args

def main():
    conf = parse_args()
    
    with open(conf.file, 'r') as fd:
        input_data = [line.strip() for line in fd.readlines()]

    evaluate(input_data)

if  __name__ == '__main__':
    main()

