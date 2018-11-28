# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum teleportation example.

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import compile, Aer

###############################################################
# Set the backend name and coupling map.
###############################################################
coupling_map = [[0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2]]
backend = Aer.get_backend("qasm_simulator")

###############################################################
# Make a quantum program for quantum teleportation.
###############################################################
q = QuantumRegister(3, "q")
c0 = ClassicalRegister(1, "c0")
c1 = ClassicalRegister(1, "c1")
c2 = ClassicalRegister(1, "c2")
qc = QuantumCircuit(q, c0, c1, c2, name="teleport")

# Prepare an initial state
qc.u3(0.3, 0.2, 0.1, q[0])

# Prepare a Bell pair
qc.h(q[1])
qc.cx(q[1], q[2])

# Barrier following state preparation
qc.barrier(q)

# Measure in the Bell basis
qc.cx(q[0], q[1])
qc.h(q[0])
qc.measure(q[0], c0[0])
qc.measure(q[1], c1[0])

# Apply a correction
qc.barrier(q)
qc.z(q[2]).c_if(c0, 1)
qc.x(q[2]).c_if(c1, 1)
qc.measure(q[2], c2[0])

###############################################################
# Execute.
# Experiment does not support feedback, so we use the simulator
###############################################################

# First version: not mapped
initial_layout = {("q", 0): ("q", 0), ("q", 1): ("q", 1),
                          ("q", 2): ("q", 2)}
qobj = compile(qc, backend=backend, coupling_map=None, shots=1024, initial_layout=initial_layout)
job = backend.run(qobj)
qobj_exp = qobj.experiments[0]

result = job.result()
print(result.get_counts(qc))

# Second version: mapped to 2x8 array coupling graph
qobj = compile(qc, backend=backend, coupling_map=coupling_map, shots=1024,initial_layout=initial_layout)
qobj_exp = qobj.experiments[0]
qobj_exp.header.compiled_circuit_qasm = ""
job = backend.run(qobj)
result = job.result()
print(result.get_counts(qc))
print(result.data(0))
# Both versions should give the same distribution
