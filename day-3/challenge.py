#!/usr/bin/env python

import argparse
import os
import pprint
import re
import sys

from collections import UserList
from dataclasses import dataclass, field
from functools import reduce
from math import inf as Infinity
from pathlib import Path
from typing import Any, Union

pp = pprint.PrettyPrinter(indent=4, width=120)
ap = pp.pprint

SYMBOL_RE = re.compile(r'[^\d.]')
NUMBER_RE = re.compile(r'\d+')

SBoard = list[list[str]]

@dataclass(kw_only=True)
class Point:
    x: int = field(default=0)
    y: int = field(default=0)

@dataclass(kw_only=True)
class DataPoint(Point):
    board: 'Board'
    
    @property
    def value(self) -> str:
        return self.board.get(self)
    
    @property
    def is_number(self) -> bool:
        return self.value.isnumeric()

    @property
    def is_dot(self) -> bool:
        return self.value == '.'
    
    @property 
    def is_symbol(self) -> bool:
        return not self.is_dot and not self.is_number

    @property
    def type(self) -> str:
        if self.is_number:
            return 'number'
        elif self.is_dot:
            return 'dot'
        else:
            return 'symbol'

class RowVectors(UserList):
    def __getitem__(self, index) -> Union['RowVector',None]:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f'RVectors{self.data.__repr__()}'

    def get(self, index) -> 'RowVector':
        if index == -1:
            return self.data[-1]
        else:
            for item in self.data:
                if index in item: 
                    return item

        raise IndexError(f'{index} not found; expected 0 < {index} < {self.max}')
    
    @property
    def first(self) -> 'RowVector':
        return self.data[0]
            
    @property
    def last(self) -> 'RowVector':
        return self.data[-1]
        
    @property 
    def min(self) -> int:
        return self.data[0].a.x

    @property
    def max(self) -> int:
        return self.data[-1].b.x
    
    def __str__(self) -> str:
        return ''.join(map(lambda v: str(v), self))
    
@dataclass(kw_only=True)
class Vector:
    a: Point
    b: Point
    
    def __contains__(self, point: Point) -> bool:
        if not isinstance(point, Point):
            point = Point(x=point, y=self.a.y)

        if all([self.a.x <= point.x and point.x <= self.b.x,
               point.y >= self.a.y and point.y <= self.b.y]):
                    # print(f"{self.a.x} <= ({point.x}) <= {self.b.x}")
                    return True
        
        return False

    @property
    def distance(self) -> Point:
        return Point(self.b.x - self.a.x, self.b.y - self.a.y)

@dataclass(kw_only=True)
class RowVector(Vector):
    board: Union['Board',None] = None
    
    @classmethod
    def create(cls: 'RowVector', board: 'Board') -> 'RowVector':
        return cls(DataPoint(board=board), DataPoint(board=board))
        
    def __repr__(self) -> str:
        return f'RowVector( range=(({self.a.x},{self.a.y}), ({self.b.x},{self.b.y})) value={self.value} )'
    
    def __str__(self) -> str:
        return self.value

    @property
    def value(self) -> str:
        return ''.join(self.board.get(self.a, self.b))

    @property
    def type(self) -> str:
        return self.a.type
    
    @property
    def is_number(self) -> bool:
        return self.a.is_number

    @property
    def is_dot(self) -> bool:
        return self.a.is_dot
    
    @property 
    def is_symbol(self) -> bool:
        return self.a.is_symbol

    @property
    def part_numbers(self) -> bool:
        if not self.is_number:
            return False
        
        x_start = self.a.x if self.a.x == 0 else self.a.x -1
        x_end   = self.b.x if self.b.x >= self.board[self.a.y].max-1 else self.b.x+1
        y_start = self.a.y if self.a.y == 0 else self.a.y - 1
        y_end   = self.a.y if self.a.y >= (len(self.board)-1) else self.a.y+1
        
        # print(f"X({x_start},{x_end}), Y({y_start},{y_end}) -> {self.value}")
        for row in range(y_start, y_end + 1):
            for column in range(x_start, x_end + 1):
                # print(f"Checking: {row},{column}")
                if DataPoint(x=column, y=row, board=self.board).is_symbol and self.is_number:
                    # print(f"found one! -> {self.value}")
                    return True
        return False
    
    @property
    def gear_ratios(self) -> bool:
        if not self.is_symbol:
            return 0
        
        if self.is_symbol and self.value != '*':
            return 0
        
        gears = set()
        
        x_start = self.a.x if self.a.x == 0 else self.a.x -1
        x_end   = self.b.x if self.b.x >= self.board[self.a.y].max-1 else self.b.x+1
        y_start = self.a.y if self.a.y == 0 else self.a.y - 1
        y_end   = self.a.y if self.a.y >= (len(self.board)-1) else self.a.y+1
        
        # print(f"X({x_start},{x_end}), Y({y_start},{y_end}) -> {self.value}")
        for row in range(y_start, y_end + 1):
            for column in range(x_start, x_end + 1):
                # print(f"Checking: {row},{column}")
                dp = DataPoint(x=column, y=row, board=self.board)
                if dp.is_number:
                    value = self.board[row].get(column).value
                    gears.add(int(value))
       
        # only counts if we have two parts connect to a star '*'
        if len(gears) == 2:
            print(f"Got Gears -> {gears}")       
            return reduce(lambda a,b: a*b, gears)

        return 0

class Board:
    def __init__(self, data):
        self._board = []        
        self._data  = data
        
        for y, row in enumerate(data):
            self._board.insert(y, RowVectors())

            for x, column in enumerate(row):
                dp = DataPoint(x=x, y=y, board=self)
                
                if x <= 0:
                    self._board[y].append(RowVector(a=dp, b=dp, board=self))
                elif self._board[y].last.type == dp.type:
                    self._board[y].last.b = dp
                else:
                    self._board[y].append(RowVector(a=dp, b=dp, board=self))
    
    def get(self, start: Point, end: Union[Point,None] = None):
        if end is None:
            end = start
        return ''.join(self._data[start.y][start.x:end.x+1])
    
    def __len__(self) -> int:
        return len(self._board)

    def __getitem__(self, index):
        return self._board[index]
    

def evaluate(board: list[list[str]]):
    board = Board(board)

    part_numbers = [int(item.value) for row in board for item in row if item.part_numbers]
    gear_ratios = [item.gear_ratios for row in board for item in row]

    for row in board:
        print(row)


    print(f"Part 1: Sum of viable numbers: {sum(part_numbers)}")
    print(f"Part 2: Sum of game Powers: {sum(gear_ratios)}")

def parse_args() -> dict[str,str]:
    def is_file(path):
        path = Path(path)
        if path.is_file() and path.exists():
            return Path(path).resolve()
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a file, or doesn't exist")

    formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=100)
    parser = argparse.ArgumentParser(formatter_class = formatter_class)

    parser.add_argument('--input-data', '-i', metavar="FILE", dest="file", help='path to input file', type=is_file, required=False, default=Path('sample.txt'))

    args, _ = parser.parse_known_args()
    
    return args

def main():
    conf = parse_args()
    
    with open(conf.file, 'r') as fd:
        input_data = [list(line.strip()) for line in fd.readlines()]

    evaluate(input_data)

if  __name__ == '__main__':
    main()

