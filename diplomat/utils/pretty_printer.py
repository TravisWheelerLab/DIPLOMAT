"""
Provides utilities for printing results to the console in a well formatted manner.
"""
import shutil
import sys
from typing import TextIO


class IndentPrinter:
    def __init__(self, output: TextIO = sys.stdout, indent: int = 4, term_width: int = None):
        self._term_width = term_width
        self._output = output
        self._indent = indent

    def __call__(self, *args, sep=' ', end='\n'):
        term_width = max(
            1, (self._term_width if(self._term_width is not None) else shutil.get_terminal_size().columns) - 2
        )

        resulting_str = sep.join(str(arg) for arg in args) + end

        lines = resulting_str.split("\n")

        for li, line in enumerate(lines):
            tab_count = 0
            for c in line:
                if(c != "\t"):
                    break
                tab_count += 1

            indent = self._indent * tab_count
            line = line[tab_count:]

            # Print the rest of the line using this many indents...
            words = line.split(" ")
            current_total = 0
            next_line = []

            for word in words:
                if((len(next_line) > 1) and ((indent + len(word) + current_total) > term_width)):
                    self._output.write((" " * indent) + (" ".join(next_line)) + "\n")
                    next_line.clear()
                    current_total = 0

                next_line.append(word)
                current_total += len(word) + 1

            if(len(next_line) > 0):
                self._output.write((" " * indent) + (" ".join(next_line)) + ("\n" if(li != len(lines) - 1) else ""))
                next_line.clear()


printer = IndentPrinter()


