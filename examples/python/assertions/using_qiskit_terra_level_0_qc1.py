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
Example showing how to use assertions in Qiskit-Terra at level 0 (novice).

It builds a bell state circuit with assertions
and runs it on the BasicAer (local Qiskit provider).

This corresponds to the first example in using_qiskit_terra_level_0.py
within basic Qiskit Terra examples.

To control the compile parameters we have provided a transpile function which can be used
as a level 1 user.

"""

# Import the Qiskit modules
from qiskit import QuantumCircuit, QiskitError
from qiskit import execute, BasicAer

# making first circuit: bell state
qc1 = QuantumCircuit(2, 2)
qc1.h(0)

# Insert a breakpoint, asserting that the 2 qubits are in a product state,
# with a critical p-value of 0.05.
breakpoint = qc1.get_breakpoint_product(0, 0, 1, 1, 0.05)

qc1.cx(0, 1)
qc1.measure([0,1], [0,1])

# setting up the backend
print("(BasicAER Backends)")
print(BasicAer.backends())

# running the breakpoint and the job
job_sim = execute([breakpoint, qc1], BasicAer.get_backend('qasm_simulator'))
result = job_sim.result()

# Show the assertion
print("Results of our " + result.get_assertion_type(breakpoint) + " Assertion:")
tup = result.get_assertion_stats(breakpoint)
print('chisq = %s\npval = %s\npassed = %s\n' % tuple(map(str,tup)))
assert ( result.get_assertion_passed(breakpoint) )

# Show the results
print("result.get_counts(qc1) = ")
print(result.get_counts(qc1))
