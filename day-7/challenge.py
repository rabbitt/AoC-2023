#!/usr/bin/env python

import argparse
import math
import os
import re
import sys

from collections import UserList
from dataclasses import dataclass, field, InitVar
from functools import reduce, _make_key
from itertools import combinations, permutations
from pathlib import Path
from threading import Lock
from typing import Union, Hashable
from enum import Enum

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
    
# Card ranks in descending order, with prime number rank values
# Using primes ensures that the product of any hand's card ranks
# will always be unique to that hand
RANKS = {
    'A': 41, 'K': 37, 'Q': 31, 'J': 29, 
    'T': 23, '9': 19, '8': 17, '7': 13, 
    '6': 11, '5': 7, '4': 5, '3': 3, '2': 2
}

PLAYER_RE = re.compile('^([AKQJT2-9]{5})\s+(\d+)')
HAND_RE = re.compile('^([AKQJT2-9]{5})')
BID_RE  = re.compile('(?:\s+(\d+))$')

ENABLE_JOKER = False

def exclude_rank(items: Union[str,list[str]]) -> list:
    if isinstance(items, str):
        items = [items]
    
    ranks = list(RANKS.keys())
    
    for item in items:
        ranks.remove(item)
    
    return ranks

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

class HandTypes(Enum):
    INDETERMINATE = -1
    HIGHEST_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    FULL_HOUSE = 4
    FOUR_OF_A_KIND = 5
    FIVE_OF_A_KIND = 6
    
    @classmethod
    def max(cls) -> int:
        return max([x.value for x in cls])

    @classmethod
    def min(cls) -> int:
        return min([x.value for x in cls])

    def __hash__(self) -> int:
        return self.value
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, HandTypes):
            return NotImplemented
        return self.value < other.value
    
    def __repr__(self) -> str:
        return f"HandTypes({str(self)})"

    def __str__(self) -> str:
        return { 
            -1: 'Indeterminate', 
            0: 'Highest Card',
            1: 'One Pair',
            2: 'Two Pair',
            3: 'Three of a Kind',
            4: 'Full House',
            5: 'Four of a Kind',
            6: 'Five of a Kind'
        }[self.value]

    def __add__(self, other: Union['HandTypes',int]) -> 'HandTypes':
        if not isinstance(other, int) and not isinstance(other, HandTypes):
            return NotImplemented('only int or HandTypes can be added to a HandType')
        elif isinstance(other, int):
            new_value = self.value + other
        elif isinstance(other, HandTypes):
            new_value = self.value + other.value
        else:
            new_value = self.value
            
        return HandTypes(new_value if new_value <= HandTypes.max() else HandTypes.max())

class LookupEntry:
    # Maps one ranking to another depending on the joker count in the hand
    # this is basically: { joker_count: { from_hand_type: to_hand_type } }
    __KIND_MAPPINGS: dict[int,dict[HandTypes,HandTypes]] = {
        0: { },
        1: {
            HandTypes.HIGHEST_CARD: HandTypes.ONE_PAIR,
            HandTypes.ONE_PAIR: HandTypes.THREE_OF_A_KIND,
            HandTypes.TWO_PAIR: HandTypes.FULL_HOUSE,
            HandTypes.THREE_OF_A_KIND: HandTypes.FOUR_OF_A_KIND,
            HandTypes.FOUR_OF_A_KIND: HandTypes.FIVE_OF_A_KIND,
        },
        2: {
            HandTypes.ONE_PAIR: HandTypes.THREE_OF_A_KIND,
            HandTypes.TWO_PAIR: HandTypes.FOUR_OF_A_KIND,
            HandTypes.FULL_HOUSE: HandTypes.FIVE_OF_A_KIND,
        },
        3: {
            HandTypes.THREE_OF_A_KIND: HandTypes.FOUR_OF_A_KIND,
            HandTypes.FULL_HOUSE: HandTypes.FIVE_OF_A_KIND,
        },
        4: {
            HandTypes.FOUR_OF_A_KIND: HandTypes.FIVE_OF_A_KIND
        },
        # If there are five jokers, we have a five of a kind already, so keep it
        5: { HandTypes.FIVE_OF_A_KIND: HandTypes.FIVE_OF_A_KIND }
    }
    
    def __init__(self, hand: Union['Hand',None] = None, rank: int = sys.maxsize, kind: HandTypes = HandTypes.INDETERMINATE):
        self._hand = hand
        self._rank = rank
        self._kind = kind
        
    def __hash__(self) -> int:
        return hash(self.hand)

    @property
    def hand(self) -> Union['Hand',None]:
        return self._hand
    
    @property
    def rank(self) -> int:
        return self._rank
    
    @property
    def kind(self) -> HandTypes:
        if ENABLE_JOKER:
            if not self._hand:
                return self._kind
            
            jokers = self._hand.count(Card('J'))
            
            if jokers > 0:
                alternate_value = self.__KIND_MAPPINGS[jokers].get(self._kind, -1)
                debug3(f"{self._hand} has {jokers} jokers; {self._kind} -> {alternate_value}")
                return HandTypes(alternate_value)

        return self._kind
    
class LookupTable(metaclass=SingletonMeta):
    data: dict[int,LookupEntry] 
    
    def __init__(self):
        # ignore args
        debug1(f"Generating Lookup Table")
        self.data = self.__generate_lookup_table()

    def get(self, key, default=LookupEntry()):
        return self.data.get(key, default)
    
    def __getitem__(self, key) -> LookupEntry:
        return self.data[key]
    
    def __contains__(self, key) -> bool:
        return key in self.data
    
    def __generate_lookup_table(self):
        # make sure our RANKS are ordered as we use that ordering below
        sorted_ranks = dict(sorted(RANKS.items(), key=lambda r: r[1], reverse=True))

        lookup_table = {}
        score_ranking = 0
        
        debug1(f" - generating five of a kind hands")
        for five_kind_card in sorted_ranks.keys():
            hand = Hand(f'{five_kind_card}' * 5)
            score_ranking += 1
            lookup_table[hand.index] = LookupEntry(hand, score_ranking, HandTypes.FIVE_OF_A_KIND)
            
        debug1(f" - generating four of a kind hands")
        for four_kind_card in sorted_ranks.keys():
            for kicker in exclude_rank(four_kind_card):
                hand_string = ''.join((four_kind_card,) * 4 + (kicker,))
                score_ranking += 1
                hand = Hand(hand_string)
                lookup_table[hand.index] = LookupEntry(hand, score_ranking, HandTypes.FOUR_OF_A_KIND)

        debug1(f" - generating full house hands")
        for full_house_card in sorted_ranks.keys():
            for pair_card in exclude_rank(full_house_card):
                hand_string = ''.join((full_house_card,) * 3 + (pair_card,) * 2)
                score_ranking += 1
                hand = Hand(hand_string)
                lookup_table[hand.index] = LookupEntry(hand, score_ranking, HandTypes.FULL_HOUSE)

        debug1(f" - generating three of a kind hands")
        for trio_card in sorted_ranks.keys():
            custom_ranks = exclude_rank(trio_card)
            for other_cards in permutations(custom_ranks, 2):
                hand_string = ''.join((trio_card, trio_card, trio_card) + other_cards)
                score_ranking += 1
                hand = Hand(hand_string)
                lookup_table[hand.index] = LookupEntry(hand, score_ranking, HandTypes.THREE_OF_A_KIND)

        debug1(f" - generating two pair hands")
        for pair_cards in permutations(sorted_ranks.keys(), 2):
            for kicker in exclude_rank(list(pair_cards)):
                hand_string = ''.join(pair_cards + pair_cards + (kicker,))
                score_ranking += 1
                hand = Hand(hand_string)
                lookup_table[hand.index] = LookupEntry(hand, score_ranking, HandTypes.TWO_PAIR)

        debug1(f" - generating pair hands")
        for pair_card in sorted_ranks.keys():
            for other_cards in permutations(exclude_rank(pair_card), 3):
                hand_string = ''.join((pair_card, pair_card) + other_cards)
                score_ranking += 1
                hand = Hand(hand_string)
                lookup_table[hand.index] = LookupEntry(hand, score_ranking, HandTypes.ONE_PAIR)
        
        debug1(f" - generating high card hands")
        for cards in permutations(sorted_ranks.keys(), 5):
            hand = Hand(''.join(cards))
            score_ranking += 1
            lookup_table[hand.index] = LookupEntry(hand, score_ranking, HandTypes.HIGHEST_CARD)
        
        return lookup_table

class Card(metaclass=SingletonMeta):
    _rank: str
    _value: int = field(init=False)
    
    def __init__(self, rank: str):
        self._rank = rank.upper()
        self._value = RANKS.get(self._rank, -1)
    
    @property
    def rank(self) -> str:
        return self._rank
    
    @property
    def value(self) -> int:
        return self._value

    @property
    def sort_rank(self) -> int:
        if ENABLE_JOKER and self is Card('J'):
            return 1
        else:
            return self._value
        
    def __repr__(self) -> str:
        return f'Card({self._rank}, {self._value})'
    
    def __str__(self) -> str:
        return self._rank 
    
    def __hash__(self) -> int:
        return self._value
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.sort_rank == other.sort_rank
    
    def __lt__(self, other: 'Card') -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.sort_rank < other.sort_rank
    
@dataclass
class Hand:
    cards: Union[list[Card], str] = field(default_factory=list)
    _index: int = field(repr=False, init=False)
    _visual: str = field(repr=False, init=False)
    
    def __post_init__(self):
        if isinstance(self.cards, str):
            self.cards = [Card(c) for c in iter(self.cards)]
        self._index = reduce(lambda a,b: a*b, [card.value for card in self.cards], 1)
        self._visual = ''.join([str(c) for c in self.cards])
    
    def __getitem__(self, index) -> Card:
        return self.cards[index] # type: ignore
    
    def __repr__(self) -> str:
        return f"Hand({self._visual})"

    def __hash__(self) -> int:
        return self._index
    
    @property
    def string_comparitor(self) -> str:
        kind = LookupTable()[self.index].kind

        if ENABLE_JOKER:
            translation = str.maketrans('TJQKA', 'A0BCD')
        else:
            translation = str.maketrans('TJQKA', 'ABCDE')

        return f"{kind.value}{self._visual.translate(translation)}"
    
    def __compare_hand__(self, other: 'Hand') -> int:
        if not isinstance(other, Hand):
            return NotImplemented

        hand1 = LookupTable()[self.index]
        hand2 = LookupTable()[other.index]
        
        if hand1.kind < hand2.kind:
            return -1
        elif hand1.kind > hand2.kind:
            return 1
        
        for idx, card1 in enumerate(self.cards):
            card2 = other.cards[idx]
            if card1.value > card2.value: # type: ignore
                return 1
            elif card1.value < card2.value: # type: ignore
                return -1

        return 0
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hand):
            return NotImplemented

        return -1 < self.__compare_hand__(other) < 1
        
    def __lt__(self, other: 'Hand') -> bool:
        if not isinstance(other, Hand):
            return NotImplemented

        return self.__compare_hand__(other) < 0
        
    def count(self, item) -> int:
        return self.cards.count(item)
    
    @property
    def index(self) -> int:
        return self._index
    
    @property
    def visual(self) -> str:
        return self._visual
    
    @property
    def value(self) -> int:
        # lower rank in our lookup table is a better hand
        return LookupTable().get(self._index).rank

@dataclass
class Player:
    hand: Hand
    bid: int
    
    def __repr__(self) -> str:
        return f'{self.hand.visual}'

    def __eq__(self, other) -> bool:
        return self.hand.string_comparitor == other.hand.string_comparitor
    
    def __lt__(self, other) -> bool:
        return self.hand.string_comparitor < other.hand.string_comparitor

SPACE_RE = re.compile('\s+')

def parse(data):
    players = []

    for line in data:
        if not line.strip():
            continue
        hand, bid = SPACE_RE.split(line)
        players.append(Player(Hand(hand), int(bid)))
    
    return players

def evaluate(data: list[str]):
    global ENABLE_JOKER 
    players = list(enumerate(sorted(parse(data), reverse=False), 1))
    bid_totals = list(map(lambda x: x[0] * x[1].bid, players))
    debug2(players)
    print(f"Part 1: result: {sum(bid_totals)}")
    
    ENABLE_JOKER = True
    players = list(enumerate(sorted(parse(data), reverse=False), 1))
    bid_totals = list(map(lambda x: x[0] * x[1].bid, players))
    debug2(players)
    print(f"Part 2: result: {sum(bid_totals)}")
    return

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
    
    with open(conf.file, 'r') as fd:
        input_data = [line.strip() for line in fd.readlines()]

    evaluate(input_data)

if  __name__ == '__main__':
    main()

