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

import time

# Import the Qiskit modules
from qiskit import QuantumCircuit
from qiskit import execute, IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor


provider = IBMQ.load_account()


# making first circuit: bell state
qc1 = QuantumCircuit(2, 2)
qc1.h(0)
qc1.cx(0, 1)
qc1.measure([0,1], [0,1])

# making another circuit: superpositions
qc2 = QuantumCircuit(2, 2)
qc2.h([0,1])
qc2.measure([0,1], [0,1])

# setting up the backend
print("(BasicAER Backends)")
print(BasicAer.backends())

# running the job
job_sim = execute([qc1, qc2], BasicAer.get_backend('qasm_simulator'))
sim_result = job_sim.result()

# Show the results
print(sim_result.get_counts(qc1))
print(sim_result.get_counts(qc2))

# see a list of available remote backends
print("\n(IBMQ Backends)")
print(IBMQ.backends())

# Compile and run on a real device backend
try:
    # select least busy available device and execute.
    least_busy_device = least_busy(provider.backends(simulator=False))
except:
    print("All devices are currently unavailable.")

print("Running on current least busy device: ", least_busy_device)

# running the job
job_exp = execute([qc1, qc2], backend=least_busy_device, shots=1024, max_credits=10)

job_monitor(job_exp)
exp_result = job_exp.result()

# Show the results
print(exp_result.get_counts(qc1))
print(exp_result.get_counts(qc2))
