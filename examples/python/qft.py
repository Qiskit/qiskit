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
from qiskit import transpile, BasicAer


###############################################################
# make the qft
###############################################################
def input_state(circ, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(j)
        circ.p(-math.pi / float(2 ** (j)), j)


def qft(circ, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cp(math.pi / float(2 ** (j - k)), j, k)
        circ.h(j)


qft3 = QuantumCircuit(5, 5, name="qft3")
qft4 = QuantumCircuit(5, 5, name="qft4")
qft5 = QuantumCircuit(5, 5, name="qft5")

input_state(qft3, 3)
qft3.barrier()
qft(qft3, 3)
qft3.barrier()
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

print(qft3)
print(qft4)
print(qft5)

print("Qasm simulator")
sim_backend = BasicAer.get_backend("qasm_simulator")
job = sim_backend.run(transpile([qft3, qft4, qft5], sim_backend), shots=1024)
result = job.result()
print(result.get_counts(qft3))
print(result.get_counts(qft4))
print(result.get_counts(qft5))
