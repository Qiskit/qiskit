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
Example showing how to use Qiskit-Terra at level 0 (novice).

This example shows the most basic way to user Terra. It builds some circuits
and runs them on both the BasicAer (local Qiskit provider) or IBMQ (remote IBMQ provider).

To control the compile parameters we have provided a transpile function which can be used 
as a level 1 user.

"""

#import time

# Import the Qiskit modules
from qiskit import QuantumCircuit, QiskitError
from qiskit import execute, BasicAer
from qiskit.circuit.asserts import Asserts
from qiskit.circuit.assertclassical import AssertClassical

# making first circuit: bell state
qc1 = QuantumCircuit(2, 2)
qc1.h(0)
#print("Asserts.StatOutputs = ")
#print(Asserts.StatOutputs)
breakpoint1 = qc1.assertclassical(0, [1], [1])
#print("AssertClassical.ExpectedValues after bkpt1 = ")
#print(AssertClassical.ExpectedValues)
qc1.cx(0, 1)
print("breakpoint1.data = ")
print(breakpoint1.data)
print("qc1.data = ")
print(qc1.data)
qc1.measure([0,1], [0,1])
print("qc1.data after measuring = ")
print(qc1.data)

# making another circuit: superpositions
qc2 = QuantumCircuit(2, 2)
qc2.h([0,1])
#breakpoint2 = qc2.assertsuperposition([0,1], [0,1])
breakpoint2 = qc2.assertclassical(0, [1], [1])
qc2.measure([0,1], [0,1])

# setting up the backend
print("(BasicAER Backends)")
print(BasicAer.backends())

# running the job
job_sim = execute([qc1, breakpoint1, qc2, breakpoint2], BasicAer.get_backend('qasm_simulator'))
sim_result = job_sim.result()
am = AssertManager()
am.stat_collect(sim_result)

"""
# Show the results
print("sim_result.get_counts(qc1) = ")
print(sim_result.get_counts(qc1))
print("sim_result.get_counts(breakpoint1) = ")
print(sim_result.get_counts(breakpoint1))
print("sim_result.get_counts(qc2) = ")
print(sim_result.get_counts(qc2))
print("sim_result.get_counts(breakpoint2) = ")
print(sim_result.get_counts(breakpoint2))

job_stats = stat_test([breakpoint1, breakpoint2], sim_result)
print("job_stats = ")
print(job_stats)
"""
