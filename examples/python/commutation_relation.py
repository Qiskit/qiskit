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

from qiskit import QuantumCircuit

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation

circuit = QuantumCircuit(5)
# Quantum Instantaneous Polynomial Time example
circuit.cx(0, 1)
circuit.cx(2, 1)
circuit.cx(4, 3)
circuit.cx(2, 3)
circuit.z(0)
circuit.z(4)
circuit.cx(0, 1)
circuit.cx(2, 1)
circuit.cx(4, 3)
circuit.cx(2, 3)
circuit.cx(3, 2)

print(circuit)

pm = PassManager()
pm.append([CommutationAnalysis(), CommutativeCancellation()])
new_circuit = pm.run(circuit)
print(new_circuit)
