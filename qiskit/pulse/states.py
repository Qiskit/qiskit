# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A attr dict is maintained by the PassManager to keep information
about the current state of the program compilation"""


class AttrDict(dict):
    """A default dictionary-like object."""

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


class Analysis(AttrDict):
    """Analysis object."""


class State(AttrDict):
    """State of the compilation.

    ..note:: Should be replaced with dataclass
    """
    def __init__(self):
        self.analysis = Analysis()
        self.pulse_program = None
        self.circuit = None
        self.lowered = None
