#!/usr/bin/env python

import argparse
import math
import os
import re
import sys

from collections import UserList
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, MutableSequence, Union, Optional, Any

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

NODES_RE = re.compile('(?i)[\da-z]{3}')

class CharacterStepper():
    _chars: list[str] = []
    _repeat: Union[int,float] = 0
    
    def __init__(self, charlist: Union[str,list[str]] = '', repeat: Union[bool,float,int] = False):
        if isinstance(charlist, str):
            characters = list(charlist)
        else:
            characters = charlist
        
        self._chars = characters
        self._repeat = repeat if repeat is math.inf else int(repeat)
                
    def __iter__(self) -> Iterator:
        i = 0
        while i < self._repeat:
            i += 1
            for char in self._chars:
                yield char
                
    def __repr__(self) -> str:
        return f"CharacterStepper({''.join(self._chars)}, repeat={self._repeat})"
        
    def __str__(self) -> str:
        return ''.join(self._chars)
    
    @property
    def chars(self) -> list[str]:
        return self._chars
    
    @property 
    def repeat(self) -> Union[float,int]:
        return self._repeat
    
    @repeat.setter
    def repeat(self, value: int) -> Union[float,int]:
        original, self._repeat = self._repeat, int(value)
        return original
    
    def append(self, charlist: Union[list[str],str]) -> 'CharacterStepper':
        if isinstance(charlist, str):
            characters = list(charlist)
        elif isinstance(charlist, MutableSequence):
            characters = charlist
            
        self._chars += characters
        
        return self

    def __add__(self, other) -> 'CharacterStepper':
        if isinstance(other, CharacterStepper):
            self.append(other.chars)
        
        self.append(str(other))
        
        return self

@dataclass(repr=True)
class Node:
    value: str = field(repr=True, default='')
    left: Optional['Node'] = field(repr=True, default=None)
    right: Optional['Node'] = field(repr=True, default=None)
    
    def __repr__(self) -> str:
        left = self.left.value if self.left else 'None'
        right = self.right.value if self.right else 'None'
        return f"Node(value={self.value}, left={left}, right={right})"

    def __str__(self) -> str:
        left = self.left.value if self.left else 'None'
        right = self.right.value if self.right else 'None'
        return f"Node(value={self.value}, left={left}, right={right})"
    
    def __hash__(self) -> int:
        return int.from_bytes(bytearray(map(ord, self.value)))
    
@dataclass(repr=True)
class Challenge:
    stepper: CharacterStepper = field(default_factory=lambda: CharacterStepper(repeat=math.inf))
    lookup: dict[str,Node] = field(default_factory=dict)

    def walk_from(self, 
        search_start: str = 'AAA', 
        search_end: str = 'ZZZ'
    ):
        steps: int = 0
        nodes: list = list(filter(lambda node: node.value.endswith(search_start), self.lookup.values()))
        
        # when we find the last node in a path, we add it this found dictionary, keyed off 
        # of the Node itself, with the value being the steps to get to it
        found: dict[Node,int] = {}
        
        for direction in self.stepper:
            if not nodes:
                break
            
            new_nodes: list = []
            for idx, node in enumerate(nodes):
                if node.value.endswith(search_end):
                    found[node] = steps
                else:
                    new_nodes.append(node.left if direction == 'L' else node.right)
                    
            nodes = new_nodes
            steps += 1
            
        return math.lcm(*found.values())
    

def parse_lines(lines: list[str]) -> Challenge:
    challenge = Challenge()
    challenge.stepper += lines.pop(0).strip()
    nodes: dict[str,Node] = challenge.lookup
    
    while lines:
        line = lines.pop(0)
        
        if not line.strip():
            continue
        
        value, left, right = NODES_RE.findall(line)
        for val in [value, left, right]:
            if val not in nodes:
                nodes[val] = Node(val)
                
        node = nodes.get(value, Node(value))
        node.left = nodes.get(left, Node(left))
        node.right = nodes.get(right, Node(right))
        
    return challenge

def evaluate(lines: list[str]):
    challenge = parse_lines(lines)
    
    part_1_steps = challenge.walk_from()
    print(f"Part 1: steps taken: {part_1_steps}")

    part_2_steps = challenge.walk_from("A", "Z")
    print(f"Part 2: steps taken: {part_2_steps}")

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

