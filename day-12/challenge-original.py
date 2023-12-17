#!/usr/bin/env python

import argparse
import math
import os
import re
import sys
import time

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache, total_ordering
from itertools import zip_longest
from pathlib import Path
from pprint import pprint as pp
from queue import Queue
from typing import Hashable, Optional, Union

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

SPACE_RE  = re.compile('\s+')
HASH_RE = re.compile(r'(#+)')

@total_ordering
class RunGroups(Iterable):
    def __init__(self, data: Optional[Union[list[int],list]] = None):
        self.data: Union[list[int],list] = list(data) if data else list()
        
    def __hash__(self) -> int:
        return hash(tuple(self.data))
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self) -> str:
        return f'RunGroups({", ".join(map(str, self.data))})'
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, (RunGroups, list, tuple)):
            return self.data == list(other)
        else:
            return NotImplemented

    def __add__(self, *args, **kwargs):
        raise NotImplemented
        pass # for linting
            
    def __iadd__(self, *args, **kwargs):
        raise NotImplemented
        pass # for linting
    
    def __sub__(self, other: 'RunGroups'):
        if not isinstance(other, RunGroups):
            return NotImplemented
        return RunGroups( [abs(a-b) for a,b in zip_longest(self.data, other.data,fillvalue=0)] )

    def __lt__(self, other: Union['RunGroups', tuple, list]) -> bool:
        data: list[int] = []
        
        if isinstance(other, RunGroups):
            data = other.data
        elif isinstance(other, list):
            data = other
        elif isinstance(other, tuple):
            data = list(other)
        else:
            return NotImplemented

        if len(self.data) > len(data):
            return False

        # make sure that RunGroups(1,2,1) > RunGroups(1,1,3), and RunGroups(1,1,1) > RunGroups(1,2,1)
        if 0 < len(self.data) <= len(data):
            max_len = min(len(self.data), len(data))
            for idx, (a, b) in enumerate(zip_longest(self.data, data, fillvalue=0)):
                if idx < (max_len-1):
                    if a > 0 and a != b: 
                        return False
                elif a > b:
                    return False
                
        return True

    def __getitem__(self, key: Union[int,slice]) -> Union[int,'RunGroups']:
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            start, stop, step = key.start, key.stop, key.step
            return RunGroups(self.data[start:stop:step])
        
        if isinstance(key, int):
            if key < 0: # Handle negative indices
                key += len(self)
           
            if key < 0 or key >= len(self):
                raise IndexError(f"index ({key}) out of range")
            
            return self.data[key]

        raise TypeError("Invalid argument type")
        
    def __setitem__(self, index: int, value: int):
        self.data[index] = int(value)

    def __mul__(self, other: int) -> 'RunGroups':
        if not isinstance(other, int):
            raise NotImplemented('RunGroups can only be multiplied by an int object')
        return RunGroups(self.data * other)
    
    @property
    def run_length(self) -> int:
        return max(sum(self) + (len(self) - 1),0)
    
    def slice(self, start: int = 0, stop: int = 0, step: int = 1) -> 'RunGroups':
        return RunGroups(self.data[start:stop:step])
    
    def append(self, value: int) -> 'RunGroups':
        self.data.append(value)
        return self

    def update(self, values: list[int]) -> 'RunGroups':
        for value in values:
            self.data.append(value)
        return self
    
    def copy(self) -> 'RunGroups':
        return RunGroups(self.data.copy())
    
    @property
    def as_list(self) -> list[int]:
        return self.data.copy()

class ReductionStrategy(ABC):
    @abstractmethod
    def reduce(self, pattern, groups):
        return NotImplemented

class SimpleReductionStrategy(ReductionStrategy):
    def reduce(self, pattern: str, groups: list[int]) -> tuple[str, list[int]]:
        # Find runs of '#' and '?' in the pattern
        found_runs = re.findall(r'(?<=\.)([#?]+)(?=\.|$)', pattern)
        reduced_pattern = pattern
        reduced_groups = groups.copy()
        
        for run in found_runs:
            for i, group in enumerate(reduced_groups):
                if len(run) == group and '#' in run:
                    # Replace the run in the pattern with '.'
                    reduced_pattern = reduced_pattern.replace(run, '.' * group, 1)
                    reduced_groups[i] = 0  # Mark this group as handled
                    break

        # Remove runs that have been set to 0
        return reduced_pattern, [g for g in reduced_groups if g != 0]

class NopReducingStrategy(ReductionStrategy):
    def reduce(self, pattern: str, groups: list[int]) -> tuple[str, list[int]]:
        return pattern, groups
    
class Record:
    def __init__(self, pattern: str, groups: list[int]):
        self._folded: bool = True

        self._original_pattern: str = pattern
        self._original_groups: RunGroups = RunGroups([int(x) for x in groups])
        
        self.reduction_strategies = [NopReducingStrategy()]

        reduced_pattern, reduced_groups = self.apply_reductions(pattern, groups)
        self._solved_folded_patterns: set[str] = set()
        
        self._folded_pattern: str = reduced_pattern
        self._unfolded_pattern: str = '?'.join([self._folded_pattern] * 5)
        
        self._folded_groups: RunGroups = RunGroups(reduced_groups)
        self._unfolded_groups: RunGroups = self._folded_groups * 5
        
        self._folded_regex: re.Pattern = re.compile('^' + re.escape(self._folded_pattern.replace('?', 'Q')).replace('Q', '.') + '$')
        self._unfolded_regex: re.Pattern = re.compile('^' + re.escape(self._unfolded_pattern.replace('?', 'Q')).replace('Q', '.') + '$')
    
    @property
    def solved_patterns(self) -> set[str]:
        patterns = set()
        for pattern in self._solved_folded_patterns:
            if self._original_pattern.replace('.', '').startswith('?'):
                pattern = pattern.replace('#', '?', 1)
            if self._original_pattern.replace('.', '').endswith('?'):
                pattern = pattern[::-1].replace('#', '?', 1)[::-1] 
            patterns.add(pattern)
        return patterns

    def apply_reductions(self, pattern: str, groups: list[int]):
        for strategy in self.reduction_strategies:
            pattern, groups = strategy.reduce(pattern, groups)
        return pattern, groups

    def __repr__(self) -> str:
        return f'Record(pattern={self.pattern} groups={self.groups} folded={self.folded})'
    
    def __hash__(self) -> int:
        return hash(self.pattern)
    
    @property
    def original_pattern(self) -> str:
        return self._original_pattern
    
    @property
    def original_groups(self) -> RunGroups:
        return self._original_groups
    
    @property
    def regex(self) -> re.Pattern:
        return self._folded_regex if self._folded else self._unfolded_regex
    
    @property
    def groups(self) -> RunGroups:
        return self._folded_groups if self._folded else self._unfolded_groups
    
    @property
    def pattern(self) -> str:
        return self._folded_pattern if self._folded else self._unfolded_pattern
    
    @property
    def folded(self) -> bool:
        return self._folded
    
    @property
    def group_size(self) -> int:
        return len(self.groups)
    
    @property
    def as_folded(self) -> 'Record':
        self._folded = True
        return self
    
    @property
    def as_unfolded(self) -> 'Record':
        self._folded = False
        return self
    
    @property
    def arrangements(self) -> list[str]:
        return list(self._arrangements_counter(self.pattern))
    
    @lru_cache(maxsize=100_000)
    def _valid_pattern_match(self, pattern: str) -> bool:
        return bool(self.regex.match(pattern))

    @lru_cache(maxsize=None)
    def _is_valid(self, pattern: str, groups: RunGroups) -> bool:
        return groups == self.groups and self._valid_pattern_match(pattern)
    
    @lru_cache(maxsize=None)
    def _runs_relatively_match(self, group: RunGroups) -> bool:
        return group < self.groups
    
    @lru_cache(maxsize=None)
    def _runs_do_not_match(self, group: RunGroups) -> bool:
        return group > self.groups
    
    @lru_cache(maxsize=None)
    def _space_needed(self, group: RunGroups) -> int:
        return self.groups.run_length - group.run_length
    
    def _arrangements_counter(self, pattern: str) -> set[str]:
        findings: set[str] = set()
        pattern_length = len(self.pattern)
        pattern = self.pattern if pattern is None else pattern
        
        stack: deque = deque([(pattern, 0, RunGroups())])

        while stack:
            pattern, index, groups = stack.popleft()
            
            if self._runs_do_not_match(groups):
                # weed out non-matches as early as possible
                continue 
            
            if self._is_valid(pattern, groups):
                findings.add(pattern)
                continue
            
            prev_char = pattern[index-1] if index > 0 else None
            
            for idx, char in enumerate(pattern[index:]):
                pos = index+idx
                space_left = pattern_length + pos
                
                if space_left < self._space_needed(groups):
                    # not enough space left to accomodated the
                    # required space for the remaining runs
                    # debug3(f" -> Not Enough Space:    {str(groups):20} {space_left:3} {pattern:>100}")
                    break
                
                if char == '#':
                    if not groups or prev_char in [None, '.']:
                        groups.append(0)
                    groups[-1] += 1
                    
                    if not self._runs_relatively_match(groups):
                        # we're not matching anymore because we either
                        # have more groups of runs than the master group of runs
                        # or we've counted longer runs than those in the master group
                        # debug3(f" -> group > self.groups: {str(groups):>30} > {str(self.groups):<30} {space_left:3} {pattern:>100}")
                        break
                
                if char == '?':
                    if self._runs_relatively_match(groups):
                        stack.append((str_replace_at(pattern, pos, '.'), pos+1, groups.copy()))
                    else: 
                        # if we don't match here, we're not going to match for the hashed version either
                        break

                    hashed_groups = groups.copy()
                    if not hashed_groups or prev_char in [None, '.']:
                        hashed_groups.append(0)

                    hashed_groups[-1] += 1
                    if self._runs_relatively_match(hashed_groups):
                        stack.append((str_replace_at(pattern, pos, '#'), pos+1, hashed_groups))

                    # we're at a fork in the road, so stop running this branch
                    # and handle the two new branches
                    break
                
                prev_char = char
            
            if self._valid_pattern_match(pattern) and groups == self.groups:
                findings.add(pattern)

        return findings

def str_replace_at(string: str, index: int, character: str) -> str:
    return string[:index] + character + string[index+1:]

def parse_lines(lines: list[str]) -> list[Record]:
    records = []
    for line in lines:
        pattern, groups = SPACE_RE.split(line)
        records.append(Record(pattern.strip(), list(map(int, groups.strip().split(',')))))
    return records

def timed_run(header: str, records: list[Record], folded: bool = True):
    total = 0
    runtimes = []
    
    if DEBUG <= 0:
        total = sum([len(record.as_folded.arrangements) if folded else len(record.as_unfolded.arrangements) for record in records])
        print(f"{header}: {total}")
        return 
    
    print(f"{header}: Starting run...")

    for idx, record in enumerate([record.as_folded if folded else record.as_unfolded for record in records], 1):
        debug1(f"\r  -> Working on record # {idx:4} ({record.original_pattern:20})", end="")
    
        start = time.perf_counter()
        count = len(record.arrangements)
        total += count
        stop = time.perf_counter()
        
        runtimes.append(stop-start)
        total_time = sum(runtimes)
        average = total_time / max(len(runtimes), 1)
        mean = sorted(runtimes)[math.ceil(len(runtimes) / 2)-1]
        
        debug1(f" ({count:6} - {int(count/(stop-start)):6}/s) - previous: {runtimes[-1]:7.4f}s - average: {average:7.4f}s - mean: {mean:7.4f}s - total: {total_time:10.4f}" + ("" * 20))
    print(f"{header} Total: {total} arrangements\n")
    
def evaluate(lines: list[str]):
    try:
        records = parse_lines(lines)
        print(f"Total Arrangements: {len(records)}")
        timed_run(f"Part 1: Sum of Folded Arrangements", records)
        timed_run(f"Part 1: Sum of Unfolded Arrangements", records, folded=False)
    except KeyboardInterrupt:
        print("")
        print("Quitting...")
        sys.exit(1)
    except Exception as e:
        raise e
    
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
    evaluate(Path(conf.file).read_text().strip().splitlines())

if  __name__ == '__main__':
    main()

