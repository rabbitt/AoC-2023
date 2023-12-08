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

def parse_lines(lines: list[str]) -> None:
    # return list(map(SPACE_RE.split, lines))
    pass

def evaluate(lines: list[str]):
    # data = parse_lines(lines)
    # debug1(data)
    
    # part_1_result: list[int] = [ line for line in data ]
    # part_2_result: list[int] = [ line for line in data ]

    # print(f"Part 1: result: {sum(part_1_result)}")
    # print(f"Part 2: result: {sum(part_2_result)}")
    pass

def parse_args() -> argparse.Namespace:
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
    evaluate(Path(conf.file).read_text().splitlines())

if  __name__ == '__main__':
    main()

