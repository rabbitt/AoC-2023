#!/usr/bin/env python

import argparse
import numpy as np
import os
import sys

from contextlib import contextmanager
from pathlib import Path

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

class Mirror:
    def __init__(self, array: list[list[str]]):
        self._mirror: np.ndarray = np.array(array)
        self._block: list[str] = [''.join(row) for row in array ]
        
        # what our tolerance is for differences between rows/columns
        # zero means the rows/columns must match exactly, greater values 
        # mean that the difference must match the tolerance precisely
        # when talking about difference, we're talking about the total
        # number of characters that differ between all rows/columns in
        # a given reflection check
        self._tolerance: int = 0 
        
    def __repr__(self) -> str:
        return f'Mirror(reflection_row={self.mirror_row} reflection_column={self.mirror_column} summary={self.summary})'
    
    @property
    def mirror(self) -> np.ndarray:
        return self._mirror
    
    @property
    def block(self) -> list[str]:
        return self._block
    
    @property
    def tolerance(self) -> int:
        return self._tolerance
    
    @tolerance.setter
    def tolerance(self, value: int):
        self._tolerance = value
    
    @contextmanager
    def tolerance_of(self, value: int):
        original_tolerance, self.tolerance = self.tolerance, value
        try:
            yield self
        finally:
            self.tolerance = original_tolerance

    @property
    def mirror_row(self) -> int:
        return self.find_reflection_point(self.mirror) 

    @property
    def mirror_column(self) -> int:
        return self.find_reflection_point(self.mirror.T) 

    def find_reflection_point(self, data: np.ndarray) -> int:
        # Note: we ignore and skip past the first row because we are only concerned with
        # pairs of data, and our slicing of [:0] would result in an empty set anyway
        for idx in range(1, data.shape[0]): 
            top_half = data[:idx, :]
            bottom_half = data[idx:, :]

            # Adjust the lengths of top and bottom halves to match the shorter one
            # we want to match expanding rows/cols of data moving out from the 
            # current evaluation point, up to our smallest length 
            min_length = min(top_half.shape[0], bottom_half.shape[0])
            top_half_reversed = top_half[-min_length:][::-1]
            bottom_half_adjusted = bottom_half[:min_length]

            # Calculate the distance between corresponding characters
            if np.sum(top_half_reversed != bottom_half_adjusted) == self.tolerance:
                # this is the index where we found an expanding set of matching
                # rows/cols. Note that it is 1 based, given that we started our
                # range from 1, allowing us to use  it in the summary calculation
                return idx

        return 0

    @property
    def summary(self) -> int:
        if self.mirror_row:
            return self.mirror_row * 100
        else:
            return self.mirror_column
        
def parse_lines(lines: list[str]) -> list[Mirror]:
    mirrors: list[Mirror] = list()
    mirror_data: list[list[str]] = list()
    
    # make sure the last line is a blank so that we
    # add the last mirror to the list of mirrors
    lines.append('') if lines[-1] else None
    
    while lines:
        line = lines.pop(0)

        if not line:
            mirrors.append(Mirror(mirror_data))
            mirror_data = list()
            continue

        mirror_data.append(list(line))
        
    return mirrors
    
def evaluate(lines: list[str]):
    mirrors = parse_lines(lines)
    debug1(mirrors)
    
    print(f"Part 1: result: {sum(b.summary for b in mirrors)}")
    
    part_2_total = 0
    for mirror in mirrors:
        with mirror.tolerance_of(1) as m:
            part_2_total += m.summary
            
    print(f"Part 2: result: {part_2_total}")

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
    evaluate(Path(conf.file).read_text().strip().splitlines())

if  __name__ == '__main__':
    main()



