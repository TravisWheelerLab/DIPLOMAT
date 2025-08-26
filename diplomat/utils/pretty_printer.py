"""
Provides utilities for printing results to the console in a well formatted manner.
"""

import shutil
import sys
from typing import TextIO


class IndentPrinter:
    """
    Create a custom printing function that matches python's print function, but also
    properly handles indented lines in the CLI by indenting each following line
    when a single line in the printout can't be displayed on a single line in the
    CLI.
    """
    def __init__(
        self, output: TextIO = sys.stdout, indent: int = 4, term_width: int = None
    ):
        """
        Create a new pretty indenting printer.

        :param output: File object to write resulting text to, defaults to stdout.
        :param indent: How many spaces to turn each tab in the input into.
        :param term_width: The width of the terminal, in characters. If None, will try to automatically determine it.
        """
        self._term_width = term_width
        self._output = output
        self._indent = indent

    def __call__(self, *args, sep=" ", end="\n"):
        """
        Print to the console.

        :param args: Objects to print.
        :param sep: The seperator to use between each object in the print-out.
        :param end: The character to place onto the end of the print-out.
        """
        term_width = max(
            1,
            (
                self._term_width
                if (self._term_width is not None)
                else shutil.get_terminal_size().columns
            )
            - 2,
        )

        resulting_str = sep.join(str(arg) for arg in args) + end

        lines = resulting_str.split("\n")

        for li, line in enumerate(lines):
            tab_count = 0
            for c in line:
                if c != "\t":
                    break
                tab_count += 1

            indent = self._indent * tab_count
            line = line[tab_count:]

            # Print the rest of the line using this many indents...
            words = line.split(" ")
            current_total = 0
            next_line = []

            for word in words:
                if (len(next_line) > 1) and (
                    (indent + len(word) + current_total) > term_width
                ):
                    self._output.write((" " * indent) + (" ".join(next_line)) + "\n")
                    next_line.clear()
                    current_total = 0

                next_line.append(word)
                current_total += len(word) + 1

            if len(next_line) > 0:
                self._output.write(
                    (" " * indent)
                    + (" ".join(next_line))
                    + ("\n" if (li != len(lines) - 1) else "")
                )
                next_line.clear()


printer = IndentPrinter()
""" Default instance of the indent printer, with the default settings. Import as print to use in a module. """