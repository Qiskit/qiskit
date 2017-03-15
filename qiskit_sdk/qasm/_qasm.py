"""
OPENQASM circuit object.

Author: Jim Challenger
"""
from ._qasmexception import QasmException
from ._qasmparser import QasmParser


class Qasm(object):
    """OPENQASM circuit object."""

    def __init__(self, filename=None, data=None):
        """Create an OPENQASM circuit object."""
        if filename is None and data is None:
            raise QasmException("Missing input file and/or data")
        if filename is not None and data is not None:
            raise QasmException("File and data must not both be"
                                + " specified initializing qasm")
        self._filename = filename
        self._data = data

    def print_tokens(self):
        """Parse and print tokens."""
        if self._filename:
            self._data = open(self._filename).read()

        qasm_p = QasmParser(self._filename)
        return qasm_p.print_tokens()

    def parse(self):
        """Parse the data."""
        if self._filename:
            self._data = open(self._filename).read()
        qasm_p = QasmParser(self._filename)
        qasm_p.parse_debug(False)
        return qasm_p.parse(self._data)
