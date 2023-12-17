#!/usr/bin/env python
 
import argparse

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Record:
    """
    Represents a record with a pattern, its run configuration, and a folding state.

    Attributes:
        pattern (str): The pattern string containing '#', '.', and '?'.
        runs (list[int]): List of integers representing the run lengths.
        folded (bool): Indicates whether the pattern is in a folded state.
    """
    
    pattern: str
    runs: list[int]
    folded: bool = True
    
    @property
    def unfolded_pattern(self) -> str:
        """
        Generates the unfolded version of the pattern.

        Returns:
            str: Unfolded pattern string.
        """
        return '?'.join([self.pattern] * 5)
    
    @property
    def unfolded_runs(self) -> list[int]:
        """
        Generates the unfolded version of the run lengths.

        Returns:
            list[int]: Unfolded run lengths.
        """
        return self.runs * 5
        
    @property
    def reference_model(self) -> str:
        """
        Creates a reference model string based on the run configuration.

        Returns:
            str: Reference model string.
        """
        runs: list[int] = self.runs if self.folded else self.unfolded_runs
        return f".{'.'.join(['#' * n for n in runs])}."

    @property
    def as_folded(self) -> 'Record':
        """
        Sets the record to a folded state.

        Returns:
            Record: The record with the folded state set. WARNING: This is NOT a copy of the Record.
        """
        self.folded = True
        return self
    
    @property
    def as_unfolded(self) -> 'Record':
        """
        Sets the record to an unfolded state.

        Returns:
            Record: The record with the unfolded state set. WARNING: This is NOT a copy of the Record.
        """
        self.folded = False
        return self
    
    @property
    def arrangements(self):
        """
        Calculates the total number of valid arrangements for the pattern using.

        This slightly modified version was largely borrowed from Moritz's solution (user clrfl on github), 
        which can be found at the link below with a really amazing explanation of how this works: 
        
            https://github.com/clrfl/AdventOfCode2023/blob/master/12/explanation.ipynb
        
        Returns:
            int: Total number of valid arrangements.
        """
        reference_model = self.reference_model
        reference_length = len(reference_model)
        test_pattern = self.pattern if self.folded else self.unfolded_pattern
        
        # Used for tracking the number of ways to reach each state
        cur_states, new_states = defaultdict(int, {0: 1}), defaultdict(int)
        
        for curr_char in test_pattern:
            for curr_pos, count in cur_states.items():
                next_pos = curr_pos + 1
                
                # Check for valid transitions based on the test character and reference model
                if next_pos < reference_length:
                    # '?' can be either '.' or '#', so we handle both possibilities. 
                    # additionally, if the CURRENT character being evaluated is the same
                    # as the NEXT STATE in the reference model, then we can move to it
                    # so we record that as well
                    if curr_char == '?' or curr_char == reference_model[next_pos]:
                        new_states[next_pos] += count
                
                # If the current character is '?' or '.' and CURRENT state in reference model is '.'
                if curr_char in ['?', '.'] and reference_model[curr_pos] == ".":
                    new_states[curr_pos] += count

            cur_states, new_states = new_states.copy(), defaultdict(int, {})
            
        # The final count is the sum of ways to reach the last and second-to-last states
        return cur_states[reference_length - 1] + cur_states[reference_length - 2]

def evaluate(data: list[Record]):
    print(f"Part 1: Sum of Arrangements: {sum(r.as_folded.arrangements for r in data)}")
    print(f"Part 2: Sum of Arrangements: {sum(r.as_unfolded.arrangements for r in data)}")

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

    patterns = []
    for line in Path(conf.file).read_text().strip().splitlines():
        pattern, runs = line.split(" ")
        patterns.append(Record(pattern, list(map(int, runs.split(',')))))

    evaluate(patterns)

if  __name__ == '__main__':
    main()

