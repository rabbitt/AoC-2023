#!/usr/bin/env python

import argparse
import os
import pprint
import re
import sys

from math import inf as Infinity
from typing import Any
from pathlib import Path

pp = pprint.PrettyPrinter(indent=4, width=120)
ap = pp.pprint

GAME_RE = re.compile(r'Game (\d+)')
SET_RE  = re.compile(r'(\d+)\s+(green|blue|red)')

MAX_RED   = 12
MAX_GREEN = 13
MAX_BLUE  = 14

def dictify(data: list[str]) -> dict[str,str]:
    games: dict[str,str] = {}
    
    for line in data:
        game, *sets = re.split(r'[:;]\s+', line)
        games[int(GAME_RE.search(game).group(1))] = [
            { color[1]: int(color[0]) for color in SET_RE.findall(set_data) } for set_data in sets
        ]
        
    return dict(sorted(games.items()))

def evaluate_part(data: list[str]):
    viable_game_ids = []
    game_powers     = []
    
    for game_id, color_sets in dictify(data).items():
        if all((color_set.get('red', 0) <= MAX_RED and 
                color_set.get('green', 0) <= MAX_GREEN and 
                color_set.get('blue', 0) <= MAX_BLUE) for color_set in color_sets):
            viable_game_ids.append(int(game_id))

        red = max(map(lambda x: x.get('red', 0), color_sets))
        green = max(map(lambda x: x.get('green', 0), color_sets))
        blue = max(map(lambda x: x.get('blue', 0), color_sets))

        game_powers.append(red * green * blue)

    print(f"Part 1: Sum of viable games: {sum(viable_game_ids)}")
    print(f"Part 2: Sum of game Powers: {sum(game_powers)}")

def parse_args() -> dict[str,str]:
    def is_file(path):
        path = Path(path)
        if path.is_file() and path.exists():
            return Path(path).resolve()
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a file, or doesn't exist")

    formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=100)
    parser = argparse.ArgumentParser(formatter_class = formatter_class)

    parser.add_argument('--input-data', '-i', metavar="FILE", dest="file", help='path to input file', type=is_file, required=True, default=Path('./input.txt'))

    args, _ = parser.parse_known_args()
    
    return args

def main():
    conf = parse_args()
    
    with open(conf.file, 'r') as fd:
        input_data = [line.strip() for line in fd.readlines()]

    evaluate_part(input_data)

if  __name__ == '__main__':
    main()

