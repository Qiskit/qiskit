# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

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
"""Gates tools."""

from qiskit import InstructionSet
from qiskit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


def attach_gate(element, quantum_register, gate, gate_class):
    """Attach a gate."""
    if isinstance(quantum_register, QuantumRegister):
        gs = InstructionSet()
        for _ in range(quantum_register.size):
            gs.add(gate)
        return gs

    element._check_qubit(quantum_register)
    return element._attach(gate_class)
