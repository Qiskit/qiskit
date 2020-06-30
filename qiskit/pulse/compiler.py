# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compiler module for pulse schedules."""

from qiskit import pulse
from qiskit.pulse.propertyset import PropertyDict


class State(PropertyDict):
    """State of the compilation.

    ..note:: Should be replaced with dataclass
    """
    def __init__(self):
        self.property_set = PropertyDict()
        self.pulse_program = None


class Compiler():
    """The default pulse compiler."""

    def __init__(self):
        self.state = State()

    @property
    def pipelines(self):
        """Return compiler pipelines"""

    @pipelines.setter
    def pipelines(self, pipeline):
        """Add compiler pipeline."""

    def compile(self, program: pulse.Program) -> pulse.Program:
        for pipeline in self.pipelines:
            program = pipeline.run(program)
