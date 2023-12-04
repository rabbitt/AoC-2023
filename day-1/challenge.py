#!/usr/bin/env python

import argparse
import os
import re
import sys

from collections import UserString
from dataclasses import dataclass
from pathlib import Path

NUMBERS = {
    "one": "1", "two": "2", "three": "3", 
    "four": "4", "five": "5", "six": "6", 
    "seven": "7", "eight": "8", "nine": "9"
}

DEBUG = os.environ.get('DEBUG') is not None

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs, file=sys.stderr)

@dataclass
class WordRun:
    '''
    Represents a run of characters representing a number in word format. Can be 
    initialized with data from a re.finditer match (span() and group() data) to
    encapsulate a given word, and later check to see if other re.Match's overlap
    with this Run.
    '''
    start: int = 0
    end:   int = 0
    word:  str = ''

    def __post_init__(self):
        self._number = NUMBERS[self.word]

    def overlaps(self, other: 'Run') -> bool:
        if self.start <= other.start:
            if self.end >= other.start:
                return True
        else:
            if other.end >= self.start:
                return True     
        return False
    __contains__ = overlaps
    
    @property
    def number(self) -> str:
        return self._number

    def add_run(self, other: 'Run') -> 'Run':
        # this assumes that 'other' has a high start number
        self.word = self.word[:(other.start - self.start)] + other.word
        self.end = self.start + len(self.word) - 1
        self._number += other.number
        
        return self

class Calibration(UserString):
    '''
    Encapsulates a line of text containing digits and number words,
    providing methods for translation, and extraction of numbers as
    digits.
    '''
    
    DIGIT_RE   = re.compile(r'\d')
    NUMBERS_RE = re.compile(f"(?=({'|'.join(NUMBERS)}))")

    @property
    def translated(self) -> str:
        '''
        Converts number words (e.g., one, two, ..., nine) to 
        their digit equivalents, factoring in overlapping words 
        (e.g., twone -> two one -> 21)
        '''
        number_word_matches = list(self.NUMBERS_RE.finditer(self.data))
        
        if not number_word_matches:
            return self
        
        translations = []

        while number_word_matches:
            match1 = number_word_matches.pop(0)
            run1 = WordRun(*match1.span(1), word=match1.group(1))
            debug(f"Run of word [{run1.word}] found")
            while number_word_matches:
                match2 = number_word_matches.pop(0)
                run2 = WordRun(*match2.span(1), word=match2.group(1))
                
                if run2 not in run1:
                    debug(f"  ~ No overlap of next word [{run2.word}] with [{run1.word}]")
                    number_word_matches.insert(0, match2)
                    break
                else:
                    debug(f"  ! Overlap of next word [{run2.word}] with [{run1.word}]", end="")
                    run1.add_run(run2)
                    debug(f" -> [{run1.word}]")
            
            debug(f"  = Adding translation for [{run1.word}] to [{run1.number}]")
            
            translations.append(run1)

        for translation in translations:
            self.data = self.data.replace(translation.word, translation.number, 1)
        
        return self

    @property
    def as_number(self) -> int:
        ''' 
        returns a number where the tens digit is the first digit in the
        string, and the ones digit is the last digit in the string. If there
        is only one digit in the string, then both the tens and ones digit 
        will likewise be the same.
        '''
        if self.DIGIT_RE.search(self.data) is not None:
            first = self.DIGIT_RE.search(self.data)[0]
            last  = self.DIGIT_RE.search(self.data[::-1])[0]
            return int(first + last)
        return 0

def evaluate(data: list[str]):
    part_1_calibrations: list[int] = [ line.as_number for line in data ]
    part_2_calibrations: list[int] = [ line.translated.as_number for line in data ]

    print(f"Part 1: Sum of calibrations: {sum(part_1_calibrations)}")
    print(f"Part 2: Sum of calibrations: {sum(part_2_calibrations)}")

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
        input_data = [Calibration(line.strip()) for line in fd.readlines()]

    evaluate(input_data)

if  __name__ == '__main__':
    main()

