#!/usr/bin/env python

import argparse
import os
import re

from pathlib import Path
from collections import UserString

class Calibration(UserString):
    NUMBERS: list[str] = [ 'one', 'two', 'three', 'four','five', 'six', 'seven', 'eight', 'nine' ]
    MAP: dict[str,str] = { num: str(idx) + num[-1] for idx, num in list(enumerate(NUMBERS, 1)) }

    DIGIT_RE   = re.compile(r'\d')
    NUMBERS_RE = re.compile(f"({'|'.join(NUMBERS)})")

    @property
    def translated(self) -> str:
        while self.NUMBERS_RE.search(self.data):
            self.data = self.NUMBERS_RE.sub(lambda m: self.MAP[m.group(1)], self.data)
        return self

    @property
    def as_number(self) -> int:
        ''' 
        returns a number where the tens digit is the first digit in the
        string, and the ones digit is the last digit in the string. If there
        is only one digit in the string, then both the tens and ones digit 
        will likewise be the same.
        '''
        if self.DIGIT_RE.search(self.data) is not None:
            first = self.DIGIT_RE.search(self.data)[0]
            last  = self.DIGIT_RE.search(self.data[::-1])[0]
            return int(first + last)
        return 0

def evaluate(data: list[str]):
    part_1_calibrations: list[int] = [ line.as_number for line in data ]
    part_2_calibrations: list[int] = [ line.translated.as_number for line in data ]

    print(f"Part 1: Sum of calibrations: {sum(part_1_calibrations)}")
    print(f"Part 2: Sum of calibrations: {sum(part_2_calibrations)}")

def parse_args() -> dict[str,str]:
    def is_file(path):
        path = Path(path)
        if path.is_file() and path.exists():
            return Path(path).resolve()
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a file, or doesn't exist")

    formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=100)
    parser = argparse.ArgumentParser(formatter_class = formatter_class)

    parser.add_argument('--input-data', '-i', metavar="FILE", dest="file", help='path to input file', type=is_file, required=True, default=Path('./input.txt'))

    args, _ = parser.parse_known_args()
    
    return args

def main():
    conf = parse_args()
    
    with open(conf.file, 'r') as fd:
        input_data = [Calibration(line.strip()) for line in fd.readlines()]

    evaluate(input_data)

if  __name__ == '__main__':
    main()

