#!/usr/bin/env python

import argparse
import os
import re
import sys

from dataclasses import dataclass
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

GAME_RE = re.compile(r'Game (\d+)')
SET_RE  = re.compile(r'(\d+)\s+(green|blue|red)')
PART_RE = re.compile(r'[:;]\s+')

MAX_RED   = 12
MAX_GREEN = 13
MAX_BLUE  = 14

@dataclass
class GameSet:
    red: int = 0
    green: int = 0
    blue: int = 0

@dataclass
class Game:
    id: int
    sets: list[GameSet]

def parse_lines(data: list[str]) -> list[Game]:
    games: list[Game] = []
    
    for line in data:
        game, *sets = PART_RE.split(line)
        
        if not GAME_RE.search(game):
            raise ValueError(f"Couldn't extract game id from game component: {game}")

        game_id = int(GAME_RE.search(game).group(1)) # type: ignore

        games.append(
            Game(id=game_id, sets=[
                GameSet(**{ 
                    color: int(count) for count, color in SET_RE.findall(set_data) 
                }) for set_data in sets
            ])
        )
        
    return sorted(games, key=lambda x: x.id)

def evaluate(lines: list[str]):
    viable_game_ids = []
    game_powers     = []

    for game in parse_lines(lines):
        if all((game_set.red <= MAX_RED and 
                game_set.green <= MAX_GREEN and 
                game_set.blue <= MAX_BLUE) for game_set in game.sets):
            viable_game_ids.append(int(game.id))

        red = max(map(lambda s: s.red, game.sets))
        green = max(map(lambda s: s.green, game.sets))
        blue = max(map(lambda s: s.blue, game.sets))

        game_powers.append(red * green * blue)

    print(f"Part 1: Sum of viable games: {sum(viable_game_ids)}")
    print(f"Part 2: Sum of game Powers: {sum(game_powers)}")

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

