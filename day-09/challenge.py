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

def calculate_differences(seq):
    return [seq[i+1] - seq[i] for i in range(len(seq) - 1)]

def predict_next(seq):
    diffs = [ seq, calculate_differences(seq) ]
    while set(diffs[-1]) != {0}:  # Check the last few items for convergence
        diffs.append(calculate_differences(diffs[-1]))
    return sum([x[-1] for x in diffs])            

def predict_prev(seq):
    return predict_next(seq[::-1])

def parse_lines(lines: list[str]) -> list[list[int]]:
    return [ [int(x) for x in SPACE_RE.split(line) ] for line in lines]

def evaluate(lines: list[str]):
    data = parse_lines(lines)

    part_1_result = sum([predict_next(x) for x in data])
    part_2_result = sum([predict_prev(x) for x in data])

    print(f"Part 1: result: {part_1_result}")
    print(f"Part 2: result: {part_2_result}")
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

