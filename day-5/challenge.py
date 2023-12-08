#!/usr/bin/env python

import argparse
import os
import re
import sys

from collections import UserList
from functools import cached_property
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from pprint import pprint
from typing import Any, Union, Iterator, Iterable, Optional 

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
SEEDS_RE = re.compile(r'(?:(\d+)\s+(\d+))')
DATA_VALIDATION_RE = re.compile(r'(\d+\s*){3}')

class Range:
    def __init__(self, start: int, stop: int):
        self.start  = start
        self.stop   = stop
        self.length = abs(self.stop - self.start)
    
    def __getitem__(self, index: int) -> int:
        if self.start <= index <= self.stop:
            return index
        raise IndexError('range object index out of range')
    
    def __contains__(self, value: int) -> bool:
        return self.start <= value < self.stop
    
    def __iter__(self) -> Iterable[int]:
        return iter(range(self.start, self.stop))

    def __repr__(self) -> str:
        return f'Range({self.start}, {self.stop})'

    def __len__(self) -> int:
        return self.length

    def __lt__(self, other) -> bool:
        return self.start < other.start

    def __le__(self, other) -> bool:
        return self.start <= other.start

    def __eq__(self, other) -> bool:
        return self.start == other.start and self.stop == other.stop

    def __ne__(self, other) -> bool:
        return not(self == other)

    def __gt__(self, other) -> bool:
        return self.start > other.start

    def __ge__(self, other) -> bool:
        return self.start >= other.start

    def delta(self, other: 'Range') -> 'Range':
        start = other.start - self.start
        stop  = (other.stop - self.stop) 
        return Range(start, stop)
        
    def overlaps(self, other: 'Range') -> bool:
        return self.start < other.stop and self.stop > other.start

    def merge(self, other: 'Range') -> 'Range':
        new_start = min(self.start, other.start)
        new_stop = max(self.stop, other.stop)
        return Range(new_start, new_stop)

class RangeMapping:
    src: Range
    dst: Range

    def __init__(self, src: Range, dst: Range):
        self.src = src
        self.dst = dst
        
    def __repr__(self) -> str:
        return f'RangeMapping(src={self.src}, dst={self.dst})'
    
    def __contains__(self, num) -> bool:
        return self.src.start <= num < self.src.stop
    
    def __getitem__(self, num: int) -> int:
        if num in self.src:
            index = (num - self.src.start) + self.dst.start
            return self.dst[index]
        return num
    
    def __len__(self) -> int:
        return len(self.src)
    
    def __lt__(self, other) -> bool:
        return self.start < other.start

    def __le__(self, other) -> bool:
        return self.start <= other.start

    def __eq__(self, other) -> bool:
        return self.start == other.start and self.stop == other.stop

    def __ne__(self, other) -> bool:
        return not(self == other)

    def __gt__(self, other) -> bool:
        return self.stop > other.stop

    def __ge__(self, other) -> bool:
        return self.stop >= other.stop

    @property
    def start(self) -> int:
        return self.src.start
    
    @property
    def stop(self) -> int:
        return self.src.stop
    
    @property
    def delta(self) -> Range:
        return self.src.delta(self.dst)
    
    def overlaps(self, other: Range) -> bool:
        return self.src.overlaps(other)

class RangeGroup(UserList):
    data: list[RangeMapping]
    
    def __iter__(self) -> Iterator:
        return iter(self.data)
    
    def __repr__(self) -> str:
        return f'RangeGroup(data={self.data})'
    
    def __getitem__(self, num: int) -> int: # type: ignore 
            for range_map in self.data:
                if num in range_map:
                    return range_map[num]
            return num

    def __len__(self) -> int:
        return sum([ len(rm) for rm in self.data ])
    
    @cached_property
    def sorted(self) -> 'RangeGroup':
        new = RangeGroup()
        new.data = sorted(self.data)
        return new
    
    def append(self, source: int, destination: int, length: int): # type: ignore
        self.data.append(RangeMapping(
            src=Range(source, source + length), 
            dst=Range(destination, destination + length)))

    def get_matching_ranges(self, search: Range):
        '''
        Given a search range, looks for overlapping RangeMappings within this 
        RangeGroup, and returns both mapped and unmapped ranges, or the original
        search range if no overlapping range mappings found for this search,
        allowing the next range group to try against the same range
        '''
        mappings = []

        # Note: Range Maps are ordered by RangeMapping.src.start, so each successive 
        # RangeMapping is further to the right on the X axis, if it were graphed.
        for range_map in self.data:
            new_stop = min(range_map.stop, search.stop) + range_map.delta.stop + 1
            
            if not range_map.overlaps(search):
                # no overlap; skip to next range_map
                debug(f"  -> (1) search: ({range_map}) -> No Overlap with {search}")
                continue
            
            # At this point, we're definitely overlapping. The question is, 
            # are we oerlapping on the left, in the middle, or on the right
            if search < range_map: # i.e., search.start < range_map.start
                # |------------------search------------------|
                # |------search------|
                #            |---------rangemap---------|
                mappings += [
                    (search.start, range_map.start - 1), 
                    (range_map.start + range_map.delta.start, new_stop)
                ]
            else:
                #       |--------search--------|
                # |------------search------------|
                # |---------rangemap---------|
                mappings += [ (search.start + range_map.delta.start, new_stop) ]

            #    |------search------|
            # |---------search---------|
            # |---------rangemap---------|
            if range_map > search: # i.e., range_map.stop > search.stop
                # If we're here, then no other rangemap is relevant, so 
                # we can return immediately, and start processing the 
                # next RangeGroup. Note: this only works because we sort
                # our range mappings within the range group
                debug(f"  -> (2) search: ({range_map}):({search}) -> {mappings}")
                return [ Range(a, b) for a,b in mappings ]
            
            search.start = range_map.stop
            
        if not mappings: 
            # if we're here, the RangeMappings didn't overlap, so then it's a 
            # straight mapping of the search start/stop to the next RangeGroup. 
            mappings = [(search.start, search.stop,)]
        
        debug(f"  -> (3) search: ({range_map}):({search}) -> {mappings}")
        return [ Range(a, b) for a,b in mappings ]

class Almanac:
    def __init__(self, data: list[str]):
        self.seeds_part_1: list[int] = list()
        self.seeds_part_2: list[Range] = list()
        self.range_groups: dict[str,RangeGroup] = dict()
                
        self.load(data)
    
    def __repr__(self) -> str:
        return f"Almanac({' '.join([f'{key}={value}' for key, value in self.__dict__.items()])})"
    
    def __merge_seeds(self, seeds: list[Range]) -> list[Range]:
        if not seeds:
            return []

        seeds = sorted(seeds, key=lambda x: x.start)
        merged_seeds = [seeds[0]]

        for rng in seeds[1:]:
            last_merged = merged_seeds[-1]
            if last_merged.overlaps(rng) or last_merged.stop == rng.start:
                merged_seeds[-1] = last_merged.merge(rng)
            else:
                merged_seeds.append(rng)

        return merged_seeds

    def get_location_by_seed_id(self, seed_id: int) -> int:
        return reduce(lambda num, range_group: range_group[num], self.range_groups.values(), seed_id)
        
    def get_location_by_range(self, seed_range: Range) -> Range:
        search_set = [seed_range]
    
        for name, range_group in self.range_groups.items():
            matches = []
            for search_range in search_set:
                debug(f"get_location_by_range: {name} -> search_range: {search_range}")
                matches += range_group.get_matching_ranges(search_range)
            
            search_set = matches
        debug(f"get_location_by_range: search_set: {search_range}")
        return min(search_set)
    
    def load_mapping(self, map_name: str, map_data: list[str]):
        ranges = []
        for line in map_data:
            ranges.append(list(map(int, SPACE_RE.split(line))))
        
        range_group = RangeGroup()
        for rng in sorted(ranges, key=lambda x: x[1]):
            # NOTE: order is important here! data points are coming in from the 
            # input in the order of "destination source length" but RangeGroup#append
            # takes the arguments in the order of "source, desintation, length"
            range_group.append(rng[1], rng[0], rng[2])
        
        self.range_groups[map_name] = range_group
        
    def load(self, data: list[str]):
        while data:
            line = data.pop(0)
            
            if not line.strip():
                 continue
             
            if 'seeds:' in line:
                line_data = line.split(': ')[1]
                self.seeds_part_1 = list(map(int, SPACE_RE.split(line_data)))

                seeds = []
                for pair in zip(self.seeds_part_1[::2], self.seeds_part_1[1::2]):
                    start = int(pair[0])
                    stop = start + int(pair[1])
                    seeds.append(Range(start, stop))
                self.seeds_part_2 = self.__merge_seeds(seeds)
                
            if 'map' in line:
                map_name = SPACE_RE.split(line)[0].replace('-', '_')
                map_data = []
                
                while data:
                    line2 = data.pop(0)
                    
                    # if we encounter a blank line, then we're done with
                    # this particular set of mapping data
                    if not line2.strip():
                        break
                    
                    map_data.append(line2)
                
                self.load_mapping(map_name, map_data)


def evaluate(data: list[str]):
    almanac = Almanac(data)

    locations = []
    for seed in almanac.seeds_part_1:
        locations.append(value := almanac.get_location_by_seed_id(seed))
    print(f"Part 1: lowest location: {min(locations)}")

    locations = []
    for seed_range in almanac.seeds_part_2:
        locations.append(almanac.get_location_by_range(seed_range).start)
    print(f"Part 2: lowest location: {min(locations)}")


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

