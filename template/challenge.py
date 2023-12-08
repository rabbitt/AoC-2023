#!/usr/bin/env python

import argparse
import os
import re
import sys

from dataclasses import dataclass
from pathlib import Path
from pprint import pprint as pp

DEBUG = int(os.environ.get('DEBUG', -1) if os.environ.get('DEBUG','').strip() else -1)

def debug(*args, **kwargs):
    level = kwargs.pop('level') if 'level' in kwargs else 0
    if DEBUG >= level:
        print(*args, **kwargs, file=sys.stderr)

debug1 = lambda *args, **kwargs: debug(*args, level=1, **kwargs)
debug2 = lambda *args, **kwargs: debug(*args, level=2, **kwargs)
debug3 = lambda *args, **kwargs: debug(*args, level=3, **kwargs)
debug4 = lambda *args, **kwargs: debug(*args, level=4, **kwargs)
debug5 = lambda *args, **kwargs: debug(*args, level=5, **kwargs)

SPACE_RE  = re.compile('\s+')

def parse(data: list[str]) -> list[tuple]:
    return map(lambda line: SPACE_RE.split(line), data)

def evaluate(data: list[str]):
    parsed_data = parse(data)
    debug1(parsed_data)
    
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

