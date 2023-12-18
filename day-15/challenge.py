#!/usr/bin/env python

import argparse
import os
import re
import sys

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, reduce
from pathlib import Path
from typing import Optional

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

FASTHASH = { x: ((x*17) % 256) for x in range(0,513) }
BOXES = { x: OrderedDict() for x in range(256) }

LABEL_OPERATION_RE = re.compile(r'(?i)([a-z]+)(?:=(\d)|-)')

class Op(Enum):
    ADD = 1
    DEL = 2

@dataclass
class Lense:
    label: str 
    length: Optional[int] = None
    
    def __hash__(self) -> int:
        return self.hash
    
    def __repr__(self) -> str:
        return f'{self.label} {self.length}'
    
    @cached_property
    def hash(self) -> int:
        return hash(self.label)
    
    @property
    def op(self) -> Op:
        if self.length:
            return Op.ADD
        else:
            return Op.DEL
    
    def power(self, box, slot) -> int:
        return box * slot * self.length
    
def calculate_focus_power(sequences: list[str]):
    labels: dict[str,Lense] = {}
    
    for sequence in sequences:
        if not sequence in labels:
            # let's only create a label/lense once per unique sequence
            label, focal_length = LABEL_OPERATION_RE.findall(sequence)[0]
            labels[sequence] = Lense(label, int(focal_length) if focal_length else None)
        
        lense = labels[sequence]
        
        if lense.op == Op.ADD:
            BOXES[lense.hash][lense.label] = lense
        elif lense.label in BOXES[lense.hash]:
            del BOXES[lense.hash][lense.label]
    
    return sum([lense.power(boxid+1, slotid) for boxid, box in BOXES.items() for slotid, lense in enumerate(box.values(), 1) ])

def hash(sequence: str):
    return reduce(lambda a,b: FASTHASH[a + ord(b)], sequence, 0)

def evaluate(sequences: list[str]):
    result = sum(hash(sequence) for sequence in sequences)
    print(f"Part 1: result: {result}")
    
    focus_power = calculate_focus_power(sequences)
    print(f"Part 2: result: {focus_power}")

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
    evaluate(Path(conf.file).read_text().strip().split(','))

if  __name__ == '__main__':
    main()

