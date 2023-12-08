#!/usr/bin/env python

import argparse
import os
import re
import sys

from functools import reduce
from collections import UserList
from dataclasses import dataclass
from pathlib import Path
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

NUMBERS_RE = re.compile('\d+')

@dataclass
class Race:
    total_time: int = 0
    distance: int = 0

    @property
    def number_of_possible_wins(self):
        # oh boy - quadratic equations. Sadly, I didn't recall how to do 
        # quadratic equations offhand, so I had to look it up. But this is
        # our basic equation:
        #     d = p*(t-p)
        # where 
        #     d is distance, 
        #     p is press_time, and 
        #     t is total_time
        # so lets calculate the midpoint, and work our way out from there
        max_press_time = self.total_time / 2
        max_distance = max_press_time * (self.total_time - max_press_time)
        if max_distance <= self.distance:
            return 0

        a = -1
        b = self.total_time
        c = -self.distance

        # if this is negative, then there aren't any roots, and so 
        # no winning distances        
        discriminant = (b**2) - (4 * a * c)
        if discriminant < 0:
            return 0

        # find the roots of: p*(t-p) = d
        sqrt_discriminant = discriminant**0.5
        p1 = (-b + sqrt_discriminant) / (2*a)
        p2 = (-b - sqrt_discriminant) / (2*a)

        # Count values of p within the range where d > our distance
        return sum(1 for p in range(int(p1), int(p2) + 1) if p * (self.total_time - p) > self.distance)
      
class LeaderBoard(UserList):
    @property
    def margin_of_error(self) -> int:
        if len(self) == 1:
            return self[0].number_of_possible_wins
        
        ways = [race.number_of_possible_wins for race in self]
        return reduce(lambda a,b: a*b, ways, ways.pop(0))

def parse_data(data, merge_numbers=False) -> LeaderBoard:
    timings: list[int] = []
    distances: list[int] = []
    leaderboard: LeaderBoard = LeaderBoard()
    
    for line in data:
        if 'Time:' in line:
            timings = list(map(int, NUMBERS_RE.findall(line)))
        if 'Distance:' in line:
            distances = list(map(int, NUMBERS_RE.findall(line)))
    
    if not merge_numbers:
        for timing, distance in zip(timings, distances):
            leaderboard.append(Race(timing, distance))
    else:
        leaderboard.append(Race(int(''.join(map(str, timings))), int(''.join(map(str, distances)))))
        
    return leaderboard
    
def evaluate(data: list[str]):
    leaderboard_part_1 = parse_data(data)
    leaderboard_part_2 = parse_data(data, merge_numbers=True)
    
    print(f"Part 1: result: {leaderboard_part_1.margin_of_error}")
    print(f"Part 2: result: {leaderboard_part_2.margin_of_error}")

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

