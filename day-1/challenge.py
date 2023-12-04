#!/usr/bin/env py

import os
import re
import sys

NUMBERS: list[str] = [ 'one', 'two', 'three', 'four','five', 'six', 'seven', 'eight', 'nine' ]
MAP: dict[str,str] = { num: str(idx) + num[-1] for idx, num in list(enumerate(NUMBERS, 1)) }

DIGIT_RE   = re.compile(r'\d')
NUMBERS_RE = re.compile(f"({'|'.join(NUMBERS)})")

def translate(string: str) -> str:
    while NUMBERS_RE.search(string):
        string = NUMBERS_RE.sub(lambda m: MAP[m.group(1)], string)
    return string

def get_number(line: str) -> int:
    if DIGIT_RE.search(line) is not None:
        first = DIGIT_RE.search(line)[0]
        last  = DIGIT_RE.search(line[::-1])[0]
        return int(first + last)
    return 0

def main():
    if len(sys.argv) <= 1:
        print(f"Usage: {os.path.basename(__file__)} path/to/data")
        sys.exit(1)

    data_file: str = sys.argv[1]
    input_data: list[str] = []

    with open(data_file, 'r') as fd:
        input_data = [ translate(line.strip()) for line in fd.readlines() ]

    calibrations: list[int] = [ get_number(line) for line in input_data ]

    print(f"Calibrations: {calibrations}")
    print(f"Sum: {sum(calibrations)}")

if  __name__ == '__main__':
    main()

