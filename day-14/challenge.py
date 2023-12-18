#!/usr/bin/env python

import argparse
import os
import numpy as np
import sys

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

from enum import Enum

class TiltDirection(Enum):
    NONE  = 0
    NORTH = 1
    EAST  = 2
    SOUTH = 3
    WEST  = 4

class Platform:
    def __init__(self, data: list[str]):
        self._board: list[str] = data
        self._array: np.ndarray = np.array([list(row) for row in data])

    def __repr__(self) -> str:
        return f'Platform(board={self._array}, current_score={self.score})'

    @property
    def board(self) -> list[str]:
        return self._board

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def score(self) -> int:
        return sum(len(self.array) - np.where(self.array == 'O')[0])

    def tilt(self, direction: TiltDirection):
        {
            TiltDirection.NORTH: self.tilt_north,
            TiltDirection.SOUTH: self.tilt_south,
            TiltDirection.EAST: self.tilt_east,
            TiltDirection.WEST: self.tilt_west,
        }[direction]()
        
    def tilt_north(self):
        for col in range(self.array.shape[1]):
            self.array[:, col] = self.barrier_sort(self.array[:, col], 'north')

    def tilt_south(self):
        for col in range(self.array.shape[1]):
            self.array[:, col] = self.barrier_sort(self.array[:, col], 'south')

    def tilt_east(self):
        for row in range(self.array.shape[0]):
            self.array[row] = self.barrier_sort(self.array[row], 'east')

    def tilt_west(self):
        for row in range(self.array.shape[0]):
            self.array[row] = self.barrier_sort(self.array[row], 'west')

    def barrier_sort(self, array, direction):
        barrier_indices = np.where(array == '#')[0]
        # Add the start and end indices for segmenting
        segment_edges = np.concatenate(([0], barrier_indices + 1, [len(array)]))

        sorted_array = np.empty_like(array)
        
        for i in range(len(segment_edges) - 1):
            start, end = segment_edges[i], segment_edges[i + 1]
            segment = array[start:end]
            
            # Exclude barriers from sorting
            non_barrier_segment = segment[segment != '#']
            
            # Sort the non-barrier elements based on direction
            count_O = np.count_nonzero(non_barrier_segment == 'O')
            count_dot = len(non_barrier_segment) - count_O
            if direction in ['north', 'west']:
                sorted_segment = np.concatenate((['O'] * count_O, ['.'] * count_dot))
            else:
                sorted_segment = np.concatenate((['.'] * count_dot, ['O'] * count_O))
            
            # Place the sorted segment back into the array, skipping barriers
            non_barrier_indices = np.where(segment != '#')[0] + start
            sorted_array[non_barrier_indices] = sorted_segment
            
            # Place the barriers back in their original positions
            sorted_array[barrier_indices[barrier_indices >= start]] = '#'

        return sorted_array
    
    def print_board(self, header: Optional[str] = None, debug_level: int = 0):
        if not DEBUG >= debug_level:
            return 
        
        if header:
            print(header)
            
        for row in self.array:
            print(''.join(row))

    def reset(self) -> 'Platform':
        self._array = np.array([list(row) for row in self._board])
        return self

    def cycle(self, total_cycles: int = 1) -> tuple[int,int]:
        history = {} 
        cycle_start = 0
        cycle_length = 0
        real_cycles = 0
        
        for i in range(total_cycles):
            for dir in [TiltDirection.NORTH, TiltDirection.WEST, TiltDirection.SOUTH, TiltDirection.EAST]:
                self.tilt(dir)
                
            arrangement = hash(self.array.tobytes())
            real_cycles += 1
            # Check if this arrangement has been seen before
            if not arrangement in history:
                history[arrangement] = i
            else:
                cycle_start = history[arrangement]
                cycle_length = i - cycle_start
                break

        # Calculate the remaining cycles
        if cycle_start:
            remaining_cycles = ((total_cycles - cycle_start) % cycle_length) - 1

            # Perform the remaining cycles
            for i in range(remaining_cycles):
                for dir in [TiltDirection.NORTH, TiltDirection.WEST, TiltDirection.SOUTH, TiltDirection.EAST]:
                    self.tilt(dir)
                real_cycles += 1

        return real_cycles, (total_cycles - real_cycles)

def evaluate(lines: list[str]):
    platform = Platform(lines)

    platform.print_board(header="Before Tilt:", debug_level=2)
    platform.tilt(TiltDirection.NORTH)
    platform.print_board("After Title:", debug_level=2)
    print(f"Part 1: Tilt Direction North: Sum of Reflector Locations: {platform.score}")

    platform.reset()
    platform.print_board(header="Before Tilt:", debug_level=2)
    real, simulated = platform.cycle(1_000_000_000)
    platform.print_board(header="After {real} cycles ({simulated} simulated)", debug_level=2)
    print(f"Part 2: 1 billion cycles: Sum of Reflector Locations: {platform.score} ({real} cycles; {simulated} simulated)")


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
    return evaluate(Path(conf.file).read_text().strip().splitlines())

if  __name__ == '__main__':
    main()

