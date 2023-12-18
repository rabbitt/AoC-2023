#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np

from abc import abstractmethod
from dataclasses import dataclass, field
from functools import _make_key
from typing import Hashable, Optional, Union
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

class SingletonMeta(type):
    _instances: dict[object,dict[Hashable,object]] = {}

    def __call__(cls, *args, **kwargs):
        hash_key = _make_key(args, kwargs, False)

        if cls not in cls._instances:
            cls._instances[cls] = {}
        
        if hash_key not in cls._instances[cls]:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls][hash_key] = instance

        return cls._instances[cls][hash_key]

@dataclass
class Point:
    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)
    
    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __repr__(self) -> str:
        return f"({self.y}, {self.x})"

    def __add__(self, other: Union['Point', tuple[int, int]]) -> 'Point':
        if isinstance(other, Point) or isinstance(other, Dir):
            return Point(self.x + other.x, self.y + other.y)
        elif isinstance(other, tuple):
            return Point(self.x + other[0], self.y + other[1])
        else:
            raise NotImplementedError(f"Can't add {other} of type {type(other)} ({other}) to Point")

    def __sub__(self, other: Union['Point', tuple[int, int]]) -> 'Point':
        if isinstance(other, Point) or isinstance(other, Dir):
            return Point(self.x - other.x, self.y - other.y)
        elif isinstance(other, tuple):
            return Point(self.x - other[0], self.y - other[1])
        else:
            raise NotImplementedError(f"Can't subtract {other} of type {type(other)} ({other}) from Point")

    def __lt__(self, other: 'Point') -> bool:
        if isinstance(other, Point) or isinstance(other, Dir):
            return (self.x, self.y) < (other.x, other.y)
        else:
            raise NotImplementedError(f"Can't compare Point with {type(other)} ({other})")

    def __le__(self, other: 'Point') -> bool:
        if isinstance(other, Point) or isinstance(other, Dir):
            return (self.x, self.y) <= (other.x, other.y)
        else:
            raise NotImplementedError(f"Can't compare Point with {type(other)} ({other})")

    def __eq__(self, other: 'Point') -> bool:
        if isinstance(other, Point) or isinstance(other, Dir):
            return self.x == other.x and self.y == other.y
        else:
            raise NotImplementedError(f"Can't compare Point with {type(other)} ({other})")

NORTH = Point( 0,-1)
SOUTH = Point( 0, 1)
EAST  = Point( 1, 0)
WEST  = Point(-1, 0)

class Dir(Point):
    @classmethod
    def north(cls) -> 'Dir':
        return Dir(0, -1)
    
    @classmethod
    def south(cls) -> 'Dir':
        return Dir(0, 1)
    
    @classmethod
    def west(cls) -> 'Dir':
        return Dir(-1, 0)
    
    @classmethod
    def east(cls) -> 'Dir':
        return Dir(1, 0)
    
    def __str__(self) -> str:
        if self.x > 0:
            return 'east'
        elif self.x < 0:
            return 'west'
        elif self.y < 0:
            return 'north'
        elif self.y > 0:
            return 'south'
        else:
            return 'unknown'

class Tile(metaclass=SingletonMeta):
    _token: str = ''
    _directions: frozenset[Dir] = field(default_factory=frozenset)
    
    def __init__(self, token: str = '', directions: Optional[frozenset[Dir]] = None):
        self._token = token
        self._directions = frozenset(directions) if directions else frozenset()

    @abstractmethod
    def heading(self, dir: Dir) -> list[Dir]:
        raise NotImplemented
    
    def __repr__(self) -> str:
        return f'{self.token}'
    
    def __hash__(self) -> int:
        return hash(self._token)
    
    @property
    def token(self) -> str:
        return self._token
    
class EmptyTile(Tile):
    def heading(self, d: Dir) -> list[Dir]:
        return [d]

class MirrorTile(Tile):
    def heading(self, d: Dir) -> list[Dir]:
        if self.token == '/':
            return [ { Dir.north(): Dir.east(), Dir.south(): Dir.west(), Dir.east(): Dir.north(), Dir.west(): Dir.south() }[d] ]
        elif self.token == '\\':
            return [ { Dir.north(): Dir.west(), Dir.south(): Dir.east(), Dir.east(): Dir.south(), Dir.west(): Dir.north() }[d] ]
        else:
            raise ValueError(f"Invalid Splitter token: {self.token}")

class SplitterTile(Tile):
    def heading(self, d: Dir) -> list[Dir]:
        if self.token == '|':
            if d in [Dir.east(), Dir.west()]:
                return [Dir.north(), Dir.south()]
            else:
                return [d]
        elif self.token == '-':
            if d in [Dir.north(), Dir.south()]:
                return [Dir.east(), Dir.west()]
            else:
                return [d]
        else:
            raise ValueError(f"Invalid Splitter token: {self.token}")

class Contraption:
    def __init__(self):
        self.board: np.ndarray = np.array([], dtype=object)
        
        # we store energeized tiles as a dict of points -> times the light has passed over the point
        self.energized_tiles: dict[Point,set[Dir]] = dict()

    def load_from_file(self, filename: str) -> None:
        with open(filename, 'r') as file:
            board_str = file.read()
        self._load_board(board_str)

    def load_from_string(self, board_str: str) -> None:
        self._load_board(board_str)

    def _load_board(self, board_str: str) -> None:
        lines = board_str.strip().split('\n')
        height, width = len(lines), max(len(line) for line in lines)
        self.board = np.empty((height, width), dtype=object)

        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char in ['/', '\\']:
                    self.board[i, j] = MirrorTile(char)
                elif char in ['-', '|']:
                    self.board[i, j] = SplitterTile(char)
                else:
                    self.board[i, j] = EmptyTile()

    def traverse_and_energize(self, start_point: Point, start_direction: Dir) -> None:
        # given that we run this multiple times,
        # we always start with a clear state
        self.energized_tiles.clear()  
        
        stack = [(start_point, start_direction)]
        
        while stack:
            point, direction = stack.pop()

            # pathfind all of the tiles that the light passes through
            if 0 <= point.x < self.board.shape[1] and 0 <= point.y < self.board.shape[0]:
                if not point in self.energized_tiles:
                    self.energized_tiles[point] = set()

                if direction in self.energized_tiles[point]:
                    continue
                
                self.energized_tiles[point].add(direction)

                tile = self.board[point.y, point.x]
                new_directions = tile.heading(direction)

                for new_dir in new_directions:
                    new_point = point + new_dir
                    if new_point != point:
                        stack.append((new_point, new_dir))

    def find_best_configuration(self) -> int:
        max_energized = 0
        rows, cols = self.board.shape
        start_configs = []

        # Iterate over the edges to add starting positions and directions
        # Top and bottom rows
        for y in [0, rows - 1]:
            for x in range(cols):
                direction = Dir.south() if y == 0 else Dir.north()
                start_configs.append((Point(x, y), direction))
                # Add additional direction for corners
                if x in [0, cols - 1]:
                    corner_dir = Dir.east() if x == 0 else Dir.west()
                    start_configs.append((Point(x, y), corner_dir))

        # Left and right columns (excluding corners already covered)
        for x in [0, cols - 1]:
            for y in range(1, rows - 1):
                direction = Dir.east() if x == 0 else Dir.west()
                start_configs.append((Point(x, y), direction))

        # Iterate over all possible starting configurations
        for start_point, start_dir in start_configs:
            self.traverse_and_energize(start_point, start_dir)
            max_energized = max(max_energized, len(self.energized_tiles))

        return max_energized

def evaluate(data: str):
    contraption = Contraption()
    contraption.load_from_string(data)
    contraption.traverse_and_energize(Point(0,0), Dir.east())
    
    print(f"Part 1: result: {len(contraption.energized_tiles)}")
    print(f"Part 2: result: {contraption.find_best_configuration()}")
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

    default_file = Path(__file__).parent.joinpath('sample.txt')
    parser.add_argument('--input-data', '-i', metavar="FILE", dest="file", help='path to input file', type=is_file, required=False, default=default_file)

    args, _ = parser.parse_known_args()
    
    return args

def main():
    conf = parse_args()
    evaluate(Path(conf.file).read_text().strip())

if  __name__ == '__main__':
    main()

