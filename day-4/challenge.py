#!/usr/bin/env python

import argparse
import re

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

SPACE_RE = re.compile(r'\s+')
GAME_RE  = re.compile(r'\s*:\s+')
SEP_RE   = re.compile(r'\s+\|\s+')

@dataclass(eq=False)
class Card:
    id: int
    matches: int = 0
    value: int = 0
    tally: int = 1

def build_deck(data: list[str]) -> dict[str,Card]:
    cards = {}
    
    for idx, row in enumerate(data,1):
        winners, guesses = map(lambda x: SPACE_RE.split(x), SEP_RE.split(GAME_RE.split(row)[1]))
        correct = len([x for x in guesses if x in winners])

        # gives us the match counts and values for part 1
        cards[idx] = Card(id=idx, 
                          matches=correct, 
                          value=2 ** (correct - 1) if correct > 0 else 0)

    return cards
    
def evaluate(data: list[str]):
    cards = build_deck(data)
    
    # gather card counts (tally) for part 2
    for card in cards.values():
        for match in [c for c in cards.values() if card.id < c.id <= (card.id + card.matches)]:
            match.tally += card.tally
    
    point_total = sum(map(lambda c: c.value, cards.values()))
    card_total  = sum([c.tally for c in cards.values()])

    print(f"Part 1: Sum of points: {point_total}")
    print(f"Part 2: Sum of cards won: {card_total}")

def parse_args() -> dict[str,str]:
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
    
    with open(conf.file, 'r') as fd:
        input_data = [line.strip() for line in fd.readlines()]

    evaluate(input_data)

if  __name__ == '__main__':
    main()

