# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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
This example has assertions added to the original teleport example.

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute

###############################################################
# Set the backend name and coupling map.
###############################################################
coupling_map = [[0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2]]
backend = BasicAer.get_backend("qasm_simulator")

###############################################################
# Make a quantum program for quantum teleportation.
###############################################################
q = QuantumRegister(3, "q")
c0 = ClassicalRegister(1, "c0")
c1 = ClassicalRegister(1, "c1")
c2 = ClassicalRegister(1, "c2")
qc = QuantumCircuit(q, c0, c1, c2, name="teleport")

breakpoints = []

# Assert a classical state of all 0's
breakpoints.append(qc.get_breakpoint_classical(q, [c0[0], c1[0], c2[0]], 0.05, 0))

# Prepare an initial state
qc.u3(0.3, 0.2, 0.1, q[0])

# Prepare a Bell pair
qc.h(q[1])
qc.cx(q[1], q[2])

# Assert not product, because it's an entangled state
breakpoints.append(qc.get_breakpoint_not_product(q[1], c1[0], q[2], c2[0], 0.05))

# Barrier following state preparation
qc.barrier(q)

# Measure in the Bell basis
qc.cx(q[0], q[1])
qc.h(q[0])

# Assert uniform of 1st qubit
breakpoints.append(qc.get_breakpoint_uniform(q[0], c0[0], 0.05))

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

print("\n\nFirst version: not mapped")
initial_layout = {q[0]: 0,
                  q[1]: 1,
                  q[2]: 2}
# Execute and show results of statistical assertion tests
job = execute(breakpoints + [qc], backend=backend, coupling_map=None, shots=1024,
                    initial_layout=initial_layout)
result = job.result()
# Show the assertion
for breakpoint in breakpoints:
    print("Results of our " + result.get_assertion_type(breakpoint) + " Assertion:")
    tup = result.get_assertion_stats(breakpoint)
    print('chisq = %s\npval = %s\npassed = %s\n' % tuple(map(str,tup)))
    assert ( result.get_assertion_passed(breakpoint) )
print(result.get_counts(qc))

print("\n\nSecond version: mapped to 2x8 array coupling graph")
job = execute(breakpoints + [qc], backend=backend, coupling_map=coupling_map, shots=1024,
              initial_layout=initial_layout)
result = job.result()
# Show the assertion
for breakpoint in breakpoints:
    print("Results of our " + result.get_assertion_type(breakpoint) + " Assertion:")
    tup = result.get_assertion_stats(breakpoint)
    print('chisq = %s\npval = %s\npassed = %s\n' % tuple(map(str,tup)))
    assert ( result.get_assertion_passed(breakpoint) )
print(result.get_counts(qc))

print("\nBoth results should give the same distribution, and therefore the same assertion results.")
