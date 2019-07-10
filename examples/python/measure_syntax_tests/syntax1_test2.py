# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum Fourier Transform examples.
"""

import math
from qiskit import QuantumCircuit
from qiskit import execute, BasicAer
from qiskit.assertions.asserts import Asserts
from qiskit.assertions.assertmanager import AssertManager

###############################################################
# make the qft
###############################################################
def input_state(circ, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(j)
        circ.u1(-math.pi/float(2**(j)), j)

def qft(circ, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi/float(2**(j-k)), j, k)
        circ.h(j)

qft3 = QuantumCircuit(5, 5, name="qft3")
qft4 = QuantumCircuit(5, 5, name="qft4")
qft5 = QuantumCircuit(5, 5, name="qft5")

input_state(qft3, 3)
qft3.barrier()
qft(qft3, 3)
qft3.barrier()

# Insert a breakpoint, asserting that the 3 measured qubits are a classical value of 1,
# with a critical p-value of 0.05.
breakpoint = qft3.assertclassical(1, 0.05, range(3), range(3))
for j in range(3):
    qft3.measure(j, j)

input_state(qft4, 4)
qft4.barrier()
qft(qft4, 4)
qft4.barrier()
for j in range(4):
    qft4.measure(j, j)

input_state(qft5, 5)
qft5.barrier()
qft(qft5, 5)
qft5.barrier()
for j in range(5):
    qft5.measure(j, j)

# setting up the backend, running the breakpoint and the job
sim_backend = BasicAer.get_backend('qasm_simulator')
job = execute([breakpoint, qft3, qft4, qft5], sim_backend, shots=1024)
result = job.result()

# We obtain a dictionary of the results from our statistical test on our breakpoint
stat_outputs = AssertManager.stat_collect(breakpoint, result)
print("Results of our statistical test:")
print(stat_outputs)

# Show the results
#print(result.get_counts(qft3))
#print(result.get_counts(qft4))
#print(result.get_counts(qft5))
