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
Instruction collection.
"""
from ._instruction import Instruction
from ._qiskiterror import QISKitError


class InstructionSet(object):
    """Instruction collection."""

    def __init__(self):
        """New collection of instructions."""
        self.instructions = set([])

    def add(self, gate):
        """Add instruction to set."""
        if not isinstance(gate, Instruction):
            raise QISKitError("attempt to add non-Instruction" +
                              " to InstructionSet")
        self.instructions.add(gate)

    def inverse(self):
        """Invert all instructions."""
        for instruction in self.instructions:
            instruction.inverse()
        return self

    def q_if(self, *qregs):
        """Add controls to all instructions."""
        for gate in self.instructions:
            gate.q_if(*qregs)
        return self

    def c_if(self, classical, val):
        """Add classical control register to all instructions."""
        for gate in self.instructions:
            gate.c_if(classical, val)
        return self
