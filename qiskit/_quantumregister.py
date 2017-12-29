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
Quantum register reference object.
"""
from ._register import Register


class QuantumRegister(Register):
    """Implement a quantum register."""

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "qreg %s[%d];" % (self.name, self.size)

    def __str__(self):
        """Return a string representing the register."""
        return "QuantumRegister(%s,%d)" % (self.name, self.size)

    def __len__(self):
        """Return a int representing the size."""
        return self.size
