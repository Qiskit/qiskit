# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum Fourier Transform examples.

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

import math
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute, Aer

###############################################################
# make the qft
###############################################################
def input_state(circ, q, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(q[j])
        circ.u1(math.pi/float(2**(j)), q[j]).inverse()


def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi/float(2**(j-k)), q[j], q[k])
        circ.h(q[j])


q = QuantumRegister(5, "q")
c = ClassicalRegister(5, "c")
qft3 = QuantumCircuit(q, c, name="qft3")
qft4 = QuantumCircuit(q, c, name="qft4")
qft5 = QuantumCircuit(q, c, name="qft5")

input_state(qft3, q, 3)
qft3.barrier()
qft(qft3, q, 3)
qft3.barrier()
for j in range(3):
    qft3.measure(q[j], c[j])

input_state(qft4, q, 4)
qft4.barrier()
qft(qft4, q, 4)
qft4.barrier()
for j in range(4):
    qft4.measure(q[j], c[j])

input_state(qft5, q, 5)
qft5.barrier()
qft(qft5, q, 5)
qft5.barrier()
for j in range(5):
    qft5.measure(q[j], c[j])

print(qft3.qasm())
print(qft4.qasm())
print(qft5.qasm())

sim_backend = Aer.get_backend('qasm_simulator')
job = execute([qft3, qft4, qft5], sim_backend, shots=1024)
result = job.result()
print(result.get_counts("qft3"))
print(result.get_counts("qft4"))
print(result.get_counts("qft5"))
