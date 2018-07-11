# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Demo basic optimization: Remove Zero Rotations and Remove Double CNOTs.

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

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
