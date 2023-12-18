#!/usr/bin/env python

import argparse
import os
import re
import sys

from shapely.geometry import Polygon, Point as PolygonPoint
from dataclasses import dataclass, field
from functools import cached_property, _make_key
from pathlib import Path
from typing import MutableSequence, Union, Callable, Optional, Mapping, Hashable, Tuple, Sequence
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

BoardSequence  = MutableSequence[MutableSequence[str]]
BoardTileCount = MutableSequence[MutableSequence[int]]

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

    def as_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __repr__(self) -> str:
        return f"({self.y}, {self.x})"

    def __add__(self, other: Union['Point', Tuple[int, int]]) -> 'Point':
        if isinstance(other, Point) or isinstance(other, Direction):
            return Point(self.x + other.x, self.y + other.y)
        elif isinstance(other, tuple):
            return Point(self.x + other[0], self.y + other[1])
        else:
            raise NotImplementedError(f"Can't add {other} of type {type(other)} ({other}) to Point")

    def __sub__(self, other: Union['Point', Tuple[int, int]]) -> 'Point':
        if isinstance(other, Point) or isinstance(other, Direction):
            return Point(self.x - other.x, self.y - other.y)
        elif isinstance(other, tuple):
            return Point(self.x - other[0], self.y - other[1])
        else:
            raise NotImplementedError(f"Can't subtract {other} of type {type(other)} ({other}) from Point")

    def __lt__(self, other: Union['Point', 'Board']) -> bool:
        if isinstance(other, Point) or isinstance(other, Direction):
            return (self.x, self.y) < (other.x, other.y)
        elif isinstance(other, Board):
            return self.x < other.width and self.y < other.height
        else:
            raise NotImplementedError(f"Can't compare Point with {type(other)} ({other})")

    def __le__(self, other: Union['Point', 'Board']) -> bool:
        if isinstance(other, Point) or isinstance(other, Direction):
            return (self.x, self.y) <= (other.x, other.y)
        elif isinstance(other, Board):
            return self.x <= other.width and self.y <= other.height
        else:
            raise NotImplementedError(f"Can't compare Point with {type(other)} ({other})")

    def __eq__(self, other: Union['Point', 'Board']) -> bool:
        if isinstance(other, Point) or isinstance(other, Direction):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, Board):
            return self.x == other.width and self.y == other.height
        else:
            raise NotImplementedError(f"Can't compare Point with {type(other)} ({other})")

NORTH = Point( 0,-1)
SOUTH = Point( 0, 1)
EAST  = Point( 1, 0)
WEST  = Point(-1, 0)

class Direction(Point):
    @classmethod
    def north(cls) -> 'Direction':
        return Direction(0, -1)
    
    @classmethod
    def south(cls) -> 'Direction':
        return Direction(0, 1)
    
    @classmethod
    def west(cls) -> 'Direction':
        return Direction(-1, 0)
    
    @classmethod
    def east(cls) -> 'Direction':
        return Direction(1, 0)
    
    @cached_property 
    def is_north(self) -> bool:
        return self.y < 0
    
    @cached_property 
    def is_south(self) -> bool:
        return self.y > 0
    
    @cached_property 
    def is_west(self) -> bool:
        return self.x < 0
    
    @cached_property 
    def is_east(self) -> bool:
        return self.x > 0
    
    @cached_property
    def perpendiculars(self) -> frozenset['Direction']:
        if self.is_east or self.is_west:
            return frozenset({Direction.north(), Direction.south()})
        elif self.is_north or self.is_south:
            return frozenset({Direction.west(), Direction.east()})
        else:
            return frozenset()
    
    @cached_property
    def opposite(self) -> 'Direction':
        if self.is_east:
            return Direction.west()
        elif self.is_west:
            return Direction.east()
        elif self.is_north:
            return Direction.south()
        elif self.is_south:
            return Direction.north()
        else:
            raise ValueError(f'Unable to determine oposite direction from {self}')
    
    @cached_property
    def other_sides(self) -> frozenset['Direction']:
        return frozenset(list(self.perpendiculars) + [self.opposite])
    
    def __str__(self) -> str:
        if self.is_east:
            return 'east'
        elif self.is_west:
            return 'west'
        elif self.is_north:
            return 'north'
        elif self.is_south:
            return 'south'
        else:
            return 'unknown'

class Tile(metaclass=SingletonMeta):
    _sigil: str = ''
    _directions: frozenset[Direction] = field(default_factory=frozenset)
    
    def __init__(self, sigil: str = '', directions: Optional[frozenset[Direction]] = None):
        self._sigil = sigil
        self._directions = frozenset(directions) if directions else frozenset()
    
    def __repr__(self) -> str:
        return f'{self.sigil}'
    
    def __hash__(self) -> int:
        return hash(self._sigil)
    
    @property
    def sigil(self) -> str:
        return self._sigil
    
    @property
    def directions(self) -> frozenset[Direction]:
        return self._directions
    
    @cached_property
    def has_north(self) -> bool:
        return any(d.is_north for d in self.directions)

    @cached_property
    def has_south(self) -> bool:
        return any(d.is_south for d in self.directions)

    @cached_property
    def has_west(self) -> bool:
        return any(d.is_west for d in self.directions)

    @cached_property
    def has_east(self) -> bool:
        return any(d.is_east for d in self.directions)

    @cached_property
    def empty(self) -> bool:
        return self.sigil == ''
    
    @cached_property
    def ground(self) -> bool:
        return self.sigil == '.'
    
    @cached_property
    def start(self) -> bool:
        return self.sigil == 'S'
    
    @cached_property
    def corner(self) -> bool:
        return self.sigil in ['F', '7', 'L', 'J']
    
    @cached_property
    def wall(self) -> bool:
        return self.sigil in ['|', '-']

CARDINAL_DIRECTIONS = frozenset({Direction.north(), Direction.south(), Direction.east(), Direction.west()})

TILE_V_PIPE = Tile('|', frozenset({Direction.north(), Direction.south()}))
TILE_H_PIPE = Tile('-', frozenset({Direction.east(),  Direction.west()}))
TILE_L_JUNCTION = Tile('L', frozenset({Direction.north(), Direction.east()}))
TILE_J_JUNCTION = Tile('J', frozenset({Direction.north(), Direction.west()}))
TILE_7_JUNCTION = Tile('7', frozenset({Direction.south(), Direction.west()}))
TILE_F_JUNCTION = Tile('F', frozenset({Direction.south(), Direction.east()}))
TILE_GROUND = Tile('.')
TILE_START = Tile('S', CARDINAL_DIRECTIONS)

CONNECTIONS: Mapping[str,Tile] = {
    TILE_V_PIPE.sigil: TILE_V_PIPE,
    TILE_H_PIPE.sigil: TILE_H_PIPE,
    TILE_L_JUNCTION.sigil: TILE_L_JUNCTION,
    TILE_J_JUNCTION.sigil: TILE_J_JUNCTION,
    TILE_7_JUNCTION.sigil: TILE_7_JUNCTION,
    TILE_F_JUNCTION.sigil: TILE_F_JUNCTION,
    TILE_GROUND.sigil: TILE_GROUND,
    TILE_START.sigil: TILE_START,
}

class Board:
    _board: Optional[BoardSequence] = field(default=None, init=False)
    _height: int = 0
    _width: int = 0
    _vboard: Optional[BoardTileCount] = None
    
    def __init__(self, 
        width: int = 0, height: int = 0, initializer: Optional[Union[Callable[[], str], str]] = None,
        *, 
        board: Optional[Union['Board', list[list[str]]]] = None
    ):
        if board is not None:
            if isinstance(board, Board):
                self._board = [row[:] for row in board.board]
            elif isinstance(board, list):
                self._board = [row[:] for row in board]
        else:
            if isinstance(initializer, str):
                self._board = [[initializer for _ in range(width)] for _ in range(height)]
            elif callable(initializer):
                self._board = [[initializer() for _ in range(width)] for _ in range(height)]
            else:
                self._board = [[' ' for _ in range(width)] for _ in range(height)]

        self._height = len(board)
        self._width = len(board[0]) if board else 0
        
        self._vboard = [[' ' for _ in range(self._width)] for _ in range(self._height)]

    def __repr__(self) -> str:
        start = self.starting_coordinates
        return f"Board(width={self._width} height={self._height} board={self._board} start=({start})"
    
    def __eq__(self, other: Union[Point, 'Board']) -> bool:
        if isinstance(other, Point):
            return (other.x, other.y) == (self.width, self.height)
        else:
            raise NotImplementedError(f"Can't compare Point with {type(other)} ({other})")
    
    def __lt__(self, other: Point) -> bool:
        if isinstance(other, Point):
            return (other.x < self.width) and (other.y < self.height)
        return NotImplemented

    def __le__(self, other: Point) -> bool:
        if isinstance(other, Point):
            return (other.x <= self.width) and (other.y <= self.height)
        return NotImplemented
    
    def __contains__(self, other: Point) -> bool:
        return other.x >= 0 and other.x <= self.width-1 and other.y >= 0 and other.y <= self.height-1
    
    @classmethod
    def from_text(cls, data: str) -> 'Board':
        return Board(board=[ re.findall(r'(.)', line) for line in data.splitlines() ])

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
    def vboard(self) -> BoardSequence:
        return self._vboard
    
    @property
    def starting_coordinates(self) -> Optional[Point]:
        for y in range(self._height):
            for x, value in enumerate(self._board[y]):
                if value == 'S':
                    return Point(x, y)
        return None

    def visit(self, point: Point) -> int:
        return self._vboard[point.y][point.x]
    
    def in_playable_space(self, point: Point) -> bool:
        return point < self
    
    def tile(self, point: Point) -> Tile:
        tile_type = self._board[point.y][point.x]
        return CONNECTIONS[tile_type]
    
    def neighbors(self, point: Point) -> list[Point]:
        current_tile = self.tile(point)
        neighbors = []
        
        for direction in current_tile.directions:
            new_point = point + direction
            if new_point in self:
                new_tile = self.tile(new_point)
                if (direction.is_north and new_tile.has_south) or \
                   (direction.is_south and new_tile.has_north) or \
                   (direction.is_west and new_tile.has_east) or \
                   (direction.is_east and new_tile.has_west):
                        neighbors.append(new_point)

        return neighbors

    def reset_vboard(self, initializer: Optional[Union[Callable[[], str], str]] = lambda: ' '):
        if isinstance(initializer, str) or isinstance(initializer, int):
            self._vboard = [[initializer for _ in range(self.width)] for _ in range(self.height)]
        elif callable(initializer):
            self._vboard = [[initializer() for _ in range(self.width)] for _ in range(self.height)]
        else:
            self._vboard = [[' ' for _ in range(self.width)] for _ in range(self.height)]
    
    @property
    def find_midpoint(self) -> int:
        start = self.starting_coordinates
        stack = [[start]]
        visited = set()
        max_length = 0
        initializer = ' '
        self.reset_vboard(initializer)

        while stack:
            paths = []
            nodes = stack.pop()
            
            while nodes:
                current = nodes.pop()
                if current in visited:
                    continue
                            
                self.vboard[current.y][current.x] = 'S' if current == start else max_length
                visited.add(current)

                for neighbor in self.neighbors(current):
                    if neighbor not in visited and self.visit(neighbor) == initializer:
                        paths.append(neighbor)
                    
            if paths:
                stack.append(paths)
                max_length += 1
            
        return max_length

    @cached_property
    def polygon_points(self) -> int:
        # get the perimeter of the polygon representing our path
        # as a series of points 
        stack = [self.starting_coordinates]
        points = {}
        
        while stack:
            current = stack.pop()
            if current in points:
                continue

            points[current] = PolygonPoint(current.x, current.y)

            for neighbor in self.neighbors(current):
                if neighbor not in points:
                    stack.append(neighbor)
        
        return list(points.values())

    @property
    def total_points_inside_polygon(self) -> int:
        # Find the bounding box of the polygon using Shapely's Polygon object
        # which is built off of numpy, so should be orderes of magnitued quicker
        # than anything I come up with on my own.
        # (docs: https://shapely.readthedocs.io/en/stable/index.html)
        
        min_x = min(int(point.x) for point in self.polygon_points)
        max_x = max(int(point.x) for point in self.polygon_points)
        min_y = min(int(point.y) for point in self.polygon_points)
        max_y = max(int(point.y) for point in self.polygon_points)

        polygon = Polygon(self.polygon_points)

        return sum(1 for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1) if polygon.contains(PolygonPoint(x,y)))
        
    def print_board(self, show_board: bool = False, show_path: bool=True):
        headers = [[f'{" ":<4}']]

        if show_board: 
            headers.append(list(range(self.width)))
        if show_path:
            headers.append(list(range(self.width)))
        if len(headers) > 2:
            headers.insert(2, [f"\t| {' ':<4}"])

        rows = []
        for y in range(self.height):
            row = [[f'{y:<4}']]
            
            if show_board:
                row.append(self.board[y])
            if show_path:
                row.append(self.vboard[y])
            if len(row) > 2:
                row.insert(2, [f"\t| {y:<4}"])
                
            rows.append(row)

        print("")
        print(''.join([' '.join([f'{char:<4}' for char in row]) for row in headers]))
        for row in rows:
            print(''.join([' '.join([f'{char:<4}' for char in chars]) for chars in row]))
        print("")

def evaluate(lines: list[str]):
    board = Board.from_lines(lines)
    
    print(f"Part 1: Path Max Midpoint Length: {board.find_midpoint}")

    if DEBUG >= 1:
        board.print_board()
        
    print(f"Part 2: Encapsulated points: {board.total_points_inside_polygon}")

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
    data = Path(conf.file).read_text().splitlines()
    if not data:
        print("no data found")
        sys.exit(1)
    evaluate(data)

if  __name__ == '__main__':
    main()

