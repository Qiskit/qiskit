# coding=utf-8
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

"""Demo basic optimization: Remove Zero Rotations and Remove Double CNOTs."""

import os
import sys

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from qiskit import QuantumProgram
from qiskit import CompositeGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.rx import RXGate


Q_SPECS = {
    "name": "Program-tutorial",
    "circuits": [{
        "name": "Circuit",
        "quantum_registers": [{
            "name": "qr",
            "size": 4
        }],
        "classical_registers": [{
            "name": "cr",
            "size": 4
        }]}],
}
Q_program = QuantumProgram(specs=Q_SPECS)
circuit = Q_program.get_circuit("Circuit")
quantum_r = Q_program.get_quantum_register("qr")
classical_r = Q_program.get_classical_register('cr')

circuit.h(quantum_r[0])
circuit.rx(0, quantum_r[0])

circuit.cx(quantum_r[0], quantum_r[1])
circuit.cx(quantum_r[0], quantum_r[1])

circuit.h(quantum_r[0])

circuit.cx(quantum_r[0], quantum_r[1])
composite_gate_1 = CompositeGate("composite1", [],
                                 [quantum_r[x] for x in range(4)])

composite_gate_1._attach(CnotGate(quantum_r[0], quantum_r[1]))
circuit._attach(composite_gate_1)

circuit.h(quantum_r[0])

composite_gate_2 = CompositeGate("composite2", [],
                                 [quantum_r[x] for x in range(4)])
composite_gate_2._attach(CnotGate(quantum_r[0], quantum_r[1]))
circuit._attach(composite_gate_2)
circuit.cx(quantum_r[0], quantum_r[1])

circuit.h(quantum_r[0])

composite_gate_3 = CompositeGate("composite3", [],
                                 [quantum_r[x] for x in range(4)])
composite_gate_3._attach(CnotGate(quantum_r[0], quantum_r[1]))
composite_gate_3._attach(CnotGate(quantum_r[0], quantum_r[2]))
circuit._attach(composite_gate_3)

circuit.h(quantum_r[0])

composite_gate_4 = CompositeGate("composite4", [],
                                 [quantum_r[x] for x in range(4)])
composite_gate_4._attach(CnotGate(quantum_r[0], quantum_r[1]))
composite_gate_4._attach(RXGate(0, quantum_r[0]))
composite_gate_4._attach(CnotGate(quantum_r[0], quantum_r[1]))
circuit._attach(composite_gate_4)

print("Removed Zero Rotations: " + str(circuit.remove_zero_rotations()))

print("Removed Double CNOTs: " + str(circuit.remove_double_cnots_once()))

# QASM from a program

QASM_source = Q_program.get_qasm("Circuit")
print(QASM_source)
