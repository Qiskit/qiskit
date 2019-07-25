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

"""
Example showing how to use assertions in Qiskit-Terra at level 0 (novice).

It builds a uniform circuit with assertions and
runs it on BasicAer (local Qiskit provider).

This corresponds to the second circuit in using_qiskit_terra_level_0.py
within the basic Qiskit Terra examples.

To control the compile parameters we have provided a transpile function which can be used
as a level 1 user.

"""

# Import the Qiskit modules
from qiskit import QuantumCircuit, QiskitError
from qiskit import execute, BasicAer
# from qiskit.assertions.asserts import Asserts

# making another circuit: uniforms
qc2 = QuantumCircuit(2, 2)
qc2.h([0,1])

# Insert a breakpoint, asserting that the 2 qubits are in a uniform state,
# with a critical p-value of 0.05.
breakpoint = qc2.get_breakpoint_uniform([0,1], [0,1], 0.05)

qc2.measure([0,1], [0,1])

# setting up the backend
print("(BasicAER Backends)")
print(BasicAer.backends())

# running the breakpoint and the job
job_sim = execute([breakpoint, qc2], BasicAer.get_backend('qasm_simulator'))
sim_result = job_sim.result()

assert ( sim_result.get_assertion_passed(breakpoint) )
# We obtain a dictionary of the results from our statistical test on our breakpoint
# stat_outputs = AssertManager.stat_collect(breakpoint, sim_result)
# print("Full results of our assertion:")
# print(stat_outputs)

# Show the results
print("sim_result.get_counts(qc2) = ")
print(sim_result.get_counts(qc2))
