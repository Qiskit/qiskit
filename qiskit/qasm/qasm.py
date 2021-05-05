# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
OPENQASM circuit object.
"""
from .exceptions import QasmError
from .qasmparser import QasmParser


class Qasm:
    """OPENQASM circuit object."""

    def __init__(self, filename=None, data=None):
        """Create an OPENQASM circuit object."""
        if filename is None and data is None:
            raise QasmError("Missing input file and/or data")
        if filename is not None and data is not None:
            raise QasmError("File and data must not both be specified" "initializing qasm")
        self._filename = filename
        self._data = data

    def return_filename(self):
        """Return the filename."""
        return self._filename

    def generate_tokens(self):
        """Returns a generator of the tokens."""
        if self._filename:
            with open(self._filename) as ifile:
                self._data = ifile.read()

        with QasmParser(self._filename) as qasm_p:
            return qasm_p.read_tokens()

    def parse(self):
        """Parse the data."""
        if self._filename:
            with open(self._filename) as ifile:
                self._data = ifile.read()

        with QasmParser(self._filename) as qasm_p:
            qasm_p.parse_debug(False)
            return qasm_p.parse(self._data)
