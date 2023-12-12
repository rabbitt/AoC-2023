#!/usr/bin/env python

import argparse
import os
import re
import sys
import time

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations
from pathlib import Path
from typing import Sequence, Union, Optional

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

BoardSequence  = Sequence[Sequence[str]]

@dataclass
class Point:
    x: int
    y: int
    
    @property
    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)
    
    def __hash__(self) -> int:
        return hash(self.as_tuple)
    
    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"
    
    def __add__(self, other: Union['Point', tuple[int, int]]) -> 'Point':
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        elif isinstance(other, tuple):
            return Point(self.x + other[0], self.y + other[1])
        else:
            raise NotImplementedError(f"Can't add {other} of type {type(other)} ({other}) to Point")
    
    def __sub__(self, other: Union['Point', tuple[int, int]]) -> 'Point':
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        elif isinstance(other, tuple):
            return Point(self.x - other[0], self.y - other[1])
        else:
            raise NotImplementedError(f"Can't subtract {other} of type {type(other)} ({other}) from Point")

@dataclass(init=False)
class Pair:
    a: Point
    b: Point

    def __init__(self, a: Union[Point,tuple[int,int]], b: Union[Point,tuple[int,int]]):
        self.a = Point(*a) if isinstance(a, tuple) else a
        self.b = Point(*b) if isinstance(b, tuple) else b
    
    def __repr__(self) -> str:
        return f'({self.a.as_tuple}, {self.b.as_tuple})'
    
    def __hash__(self) -> int:
        return hash(self.a.as_tuple + self.b.as_tuple)
    
    @property
    def xrange(self):
        return range(min(self.a.x, self.b.x), max(self.a.x, self.b.x))

    @property
    def yrange(self):
        return range(min(self.a.y, self.b.y), max(self.a.y, self.b.y))

    @property
    def distance(self):
        return abs(self.b.x - self.a.x) + abs(self.b.y - self.a.y)

    @property
    def bounding_box(self) -> 'Pair':
       return Pair(
           Point(min(self.a.x, self.b.x), min(self.a.y, self.b.y)),
           Point(max(self.a.x, self.b.x), max(self.a.y, self.b.y)),
       ) 
            
    def intersects(self, *, cols: set = set(), rows: set = set()) -> tuple[set,set]:
        return len(set(self.xrange).intersection(cols)), len(set(self.yrange).intersection(rows))

class Board:
    _board: Optional[BoardSequence] = None
    _height: int = 0
    _width: int = 0
    
    _galaxies: set[Point] = set()
    _empty_rows: set = set()
    _empty_cols: set = set()
    
    def __init__(self, board: BoardSequence):
        self._board = [row[:] for row in board]
        self._height = len(board)
        self._width = len(board[0]) if board else 0
        self._empty_rows = set(range(self._height))
        self._empty_cols = set(range(self._width))

        self.find_galaxies_and_empty_space()
        

    def __repr__(self) -> str:
        return f"Board(width={self._width} height={self._height} board={self._board}"
    
    @classmethod
    def from_lines(cls, data: list[str]) -> 'Board':    
        return Board(board=[ re.findall(r'(.)', line) for line in data ])

    @property
    def board(self) -> BoardSequence:
        return self._board
    
    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height
    
    @property
    def galaxies(self) -> list[Point]:
        return self._galaxies

    @property
    def empty_rows(self) -> list[int]:
        return self._empty_rows
    
    @property
    def empty_columns(self) -> list[int]:
        return self._empty_cols

    @cached_property
    def galaxy_pairs(self):
        return [Pair(*p) for p in combinations(self.galaxies, 2)]
    
    def find_galaxies_and_empty_space(self):
        # We can save some time here and find both empty space and 
        # empty rows and columns at the same time. Note: we start
        # with empty_cols and empty_rows already filled with all
        # of the possible values the could have, and then simply
        # remove them as we find intersecting galaxies.
        for y in range(self._height):
            for x, value in enumerate(self._board[y]):
                if value == '#':
                    self._galaxies.add(Point(x, y))
                    self._empty_cols -= {x}
                    self._empty_rows -= {y}
                    
    def solve(self, part="Part 1", expansion_rate=2) -> int:
        total = 0

        for pair in self.galaxy_pairs:
            # get bounding box for galaxy pair so that we can expand it
            # based on the intersecting x and y expansion zones
            box = pair.bounding_box

            # get the number of rows/columns that intersect with expanding space
            x_multiplier, y_multiplier = box.intersects(cols=self.empty_columns, rows=self.empty_rows)

            box.b.x += ((expansion_rate - 1) * x_multiplier) 
            box.b.y += ((expansion_rate - 1) * y_multiplier)

            # now calculate the distance from the top left 
            # corner to the bottom right corner by calculating
            # the Manhattan Distance: abs(a.x - b.x) + abs(a.y - b.y)
            total += box.distance

        return total
                
    def print_board(self):
        width = len(self.board[0]) if len(self.board) > 0 else 0
        col_width = len(str(width))
        
        print(' '.join([f'{x:>{col_width}}' for x in ([' '] + list(range(width)))]))
        
        for idx, row in enumerate(self.board):
            print(' '.join([f'{x:>{col_width}}' for x in ([str(idx)]+row)]))

@contextmanager
def benchmark(display: str=' > runtime: {delta:0.4}s'):
    try:
        start = time.perf_counter()
        yield
    finally:
        stop = time.perf_counter()
    delta = stop - start
    print(display.format(delta=delta))

def evaluate(lines: list[str]):
    board = Board.from_lines(lines)

    # board.print_board()
    print(f"Total Galaxies: {len(board.galaxies)}")
    print(f"Galaxies Pairs: {len(board.galaxy_pairs)}")
    
    with benchmark():
        print(f"\nPart 1: Shortest Paths with Expansion=2: {board.solve(part='Part 1', expansion_rate=2)}")
    with benchmark():
        print(f"\nPart 2: Shortest Paths with Expansion=1M: {board.solve(part='Part 2', expansion_rate=1_000_000)}")

def parse_args() -> argparse.Namespace:
    def is_file(path):
        path = Path(path)
        if path.is_file() and path.exists():
            return Path(path).resolve()
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a file, or doesn't exist")

    formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=100)
    parser = argparse.ArgumentParser(formatter_class = formatter_class)

    default_file = Path(__file__).parent.joinpath('sample.txt')
    parser.add_argument('--input-data', '-i', metavar="FILE", dest="file", help='path to input file', type=is_file, required=False, default=default_file)

    args, _ = parser.parse_known_args()
    
    return args

def main():
    conf = parse_args()
    data = Path(conf.file).read_text().splitlines()
    if not data:
        print("no data found")
        sys.exit(1)
    evaluate(data)

if  __name__ == '__main__':
    main()


