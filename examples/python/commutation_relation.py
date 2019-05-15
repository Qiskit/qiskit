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

from qiskit import *

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation

qr = QuantumRegister(5, 'qr')
circuit = QuantumCircuit(qr)
# Quantum Instantaneous Polynomial Time example
circuit.cx(qr[0], qr[1])
circuit.cx(qr[2], qr[1])
circuit.cx(qr[4], qr[3])
circuit.cx(qr[2], qr[3]) 
circuit.z(qr[0])
circuit.z(qr[4])
circuit.cx(qr[0], qr[1])
circuit.cx(qr[2], qr[1])
circuit.cx(qr[4], qr[3])
circuit.cx(qr[2], qr[3]) 
circuit.cx(qr[3], qr[2]) 

print(circuit.draw())

pm = PassManager()
pm.append([CommutationAnalysis(), CommutativeCancellation()])
new_circuit=pm.run(circuit)
print(new_circuit.draw())
