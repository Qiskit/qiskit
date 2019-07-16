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
Quantum teleportation example.

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute
from qiskit.assertions.asserts import Asserts
from qiskit.assertions.assertmanager import AssertManager

###############################################################
# Set the backend name and coupling map.
###############################################################
coupling_map = [[0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2]]
backend = BasicAer.get_backend("qasm_simulator")

###############################################################
# Make a quantum program for quantum teleportation.
###############################################################
q = QuantumRegister(3, "q")
c = ClassicalRegister(3, "c0")
qc = QuantumCircuit(q, c, name="teleport")

# Assert a classical state of all 0's
breakpoint1 = qc.assertclassical(0, 0.05, q, c)

qc.measure(q, c)

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

qc.measure(q[0], c[0])
qc.measure(q[1], c[1])

###############################################################
# Execute.
# Experiment does not support feedback, so we use the simulator
###############################################################

# First version: not mapped
initial_layout = {q[0]: 0,
                  q[1]: 1,
                  q[2]: 2}
job = execute([breakpoint1, qc], backend=backend, coupling_map=None, shots=1024,
                    initial_layout=initial_layout)
result = job.result()
stat_outputs = AssertManager.stat_collect(breakpoint1, result)
print("Full results of our assertion, run with no coupling map:")
print(stat_outputs)
print(result.get_counts(qc))
print()

# Second version: mapped to 2x8 array coupling graph
job = execute([breakpoint1, qc], backend=backend, coupling_map=coupling_map, shots=1024,
              initial_layout=initial_layout)
result = job.result()
stat_outputs = AssertManager.stat_collect(breakpoint1, result)
print("Full results of our assertion, run with a coupling map:")
print(stat_outputs)
print(result.get_counts(qc))

print("\nBoth versions should give the same distribution, and therefore the same assertion results.")
