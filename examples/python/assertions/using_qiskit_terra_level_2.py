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
Example showing how to use Qiskit at level 2 (advanced).

This example shows how an advanced user interacts with Terra.
It builds some circuits and transpiles them with the pass_manager.
"""

import pprint, time

# Import the Qiskit modules
from qiskit import IBMQ, BasicAer
from qiskit import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.extensions import SwapGate
from qiskit.compiler import assemble
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor

from qiskit.transpiler import PassManager
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import Decompose
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import LookaheadSwap

# from qiskit.assertions.asserts import Asserts

IBMQ.load_accounts()

breakpoints = []

# Making first circuit: bell state
qc1 = QuantumCircuit(4, 4)
qc1.h(0)
qc1.cx(0, 1)
breakpoints.append(qc1.get_breakpoint_not_product(0, 0, 1, 1, 0.05))
breakpoints.append(qc1.get_breakpoint_not_uniform([0,1], [0,1], 0.05))
qc1.measure([0,1], [0,1])

# Making another circuit: GHZ State
qc2 = QuantumCircuit(4, 4)
qc2.h(0)
qc2.cx(0, 1)
qc2.cx(0, 2)
qc2.cx(0, 3)
breakpoints.append(qc2.get_breakpoint_not_product([0,1], [0,1], [2,3], [2,3], 0.05))
breakpoints.append(qc2.get_breakpoint_not_uniform([0,1,2,3], [0,1,2,3], 0.05))
qc2.measure([0,1,2,3], [0,1,2,3])

# Setting up the backend
print("(Aer Backends)")
for backend in BasicAer.backends():
    print(backend.status())
qasm_simulator = BasicAer.get_backend('qasm_simulator')


# Compile and run the circuit on a real device backend
# See a list of available remote backends
print("\n(IBMQ Backends)")
for backend in IBMQ.backends():
    print(backend.status())

try:
    # select least busy available device and execute.
    least_busy_device = least_busy(IBMQ.backends(simulator=False))
except:
    print("All devices are currently unavailable.")

print("Running on current least busy device: ", least_busy_device)


# making a pass manager to compile the circuits
coupling_map = CouplingMap(least_busy_device.configuration().coupling_map)
print("coupling map: ", coupling_map)

pm = PassManager()

# Use the trivial layout
pm.append(TrivialLayout(coupling_map))

# Extend the the dag/layout with ancillas using the full coupling map
pm.append(FullAncillaAllocation(coupling_map))
pm.append(EnlargeWithAncilla())

# Swap mapper
pm.append(LookaheadSwap(coupling_map))

# Expand swaps
pm.append(Decompose(SwapGate))

# Simplify CXs
pm.append(CXDirection(coupling_map))

# unroll to single qubit gates
pm.append(Unroller(['u1', 'u2', 'u3', 'id', 'cx']))
qc1_new = pm.run(qc1)
qc2_new = pm.run(qc2)
breakpoints_new = [pm.run(breakpoint) for breakpoint in breakpoints]

print("Bell circuit before passes:")
print(qc1)
print("Bell circuit after passes:")
print(qc1_new)
print("Superposition circuit before passes:")
print(qc2)
print("Superposition circuit after passes:")
print(qc2_new)

# Assemble the two circuits into a runnable qobj
qobj = assemble(breakpoints_new + [qc1_new, qc2_new], shots=1000)

# Running qobj on the simulator
print("Running on simulator:")
sim_job = qasm_simulator.run(qobj)

# Getting the result
sim_result=sim_job.result()

# Show assertion results
assert ( sim_result.get_assertion_passed(breakpoints[0]) )
assert ( sim_result.get_assertion_passed(breakpoints[1]) )
assert ( sim_result.get_assertion_passed(breakpoints[2]) )
assert ( sim_result.get_assertion_passed(breakpoints[3]) )
# stat_outputs = AssertManager.stat_collect(breakpoints, sim_result)
# print("Full assertion results:")
# print(stat_outputs)

# Show the results
print(sim_result.get_counts(qc1))
print(sim_result.get_counts(qc2))

# Running the job.
print("Running on device:")
exp_job = least_busy_device.run(qobj)

job_monitor(exp_job)
exp_result = exp_job.result()

# Show assertion results
assert ( sim_result.get_assertion_passed(breakpoints[0]) )
assert ( sim_result.get_assertion_passed(breakpoints[1]) )
assert ( sim_result.get_assertion_passed(breakpoints[2]) )
assert ( sim_result.get_assertion_passed(breakpoints[3]) )
# stat_outputs = AssertManager.stat_collect(breakpoints, exp_result)
# print("Full assertion results:")
# print(stat_outputs)

# Show the results
print(exp_result.get_counts(qc1))
print(exp_result.get_counts(qc2))
