# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
OPENQASM circuit object.
"""
from ._qasmerror import QasmError
from ._qasmparser import QasmParser


class Qasm(object):
    """OPENQASM circuit object."""

    def __init__(self, filename=None, data=None):
        """Create an OPENQASM circuit object."""
        if filename is None and data is None:
            raise QasmError("Missing input file and/or data")
        if filename is not None and data is not None:
            raise QasmError("File and data must not both be specified"
                            "initializing qasm")
        self._filename = filename
        self._data = data

    def get_filename(self):
        """Return the filename."""
        return self._filename

    def get_tokens(self):
        """Returns a generator of the tokens."""
        if self._filename:
            with open(self._filename) as ifile:
                self._data = ifile.read()

        with QasmParser(self._filename) as qasm_p:
            return qasm_p.get_tokens()

    def parse(self):
        """Parse the data."""
        if self._filename:
            with open(self._filename) as ifile:
                self._data = ifile.read()

        with QasmParser(self._filename) as qasm_p:
            qasm_p.parse_debug(False)
            return qasm_p.parse(self._data)
