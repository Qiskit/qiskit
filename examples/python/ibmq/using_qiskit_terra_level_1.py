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
Example showing how to use Qiskit at level 1 (intermediate).

This example shows how an intermediate user interacts with Terra.
It builds some circuits and transpiles them with transpile options.
It then makes a qobj object which is just a container to be run on a backend.
The same qobj can be submitted to many backends (as shown).
It is the user's responsibility to make sure it can be run (i.e. it conforms
to the restrictions of the backend, if any).
This is useful when you want to compare the same
circuit on different backends without recompiling the whole circuit,
or just want to change some runtime parameters.

To control the passes that transform the circuit, we have a pass manager
for the level 2 user.
"""

# Import the Qiskit modules
from qiskit import IBMQ, BasicAer
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile, assemble
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor

provider = IBMQ.load_account()

# Making first circuit: bell state
qc1 = QuantumCircuit(2, 2, name="bell")
qc1.h(0)
qc1.cx(0, 1)
qc1.measure([0,1], [0,1])

# Making another circuit: superpositions
qc2 = QuantumCircuit(2, 2, name="superposition")
qc2.h([0,1])
qc2.measure([0,1], [0,1])

# Setting up the backend
print("(Aer Backends)")
for backend in BasicAer.backends():
    print(backend.status())
qasm_simulator = BasicAer.get_backend('qasm_simulator')


# Compile and run the circuit on a real device backend
# See a list of available remote backends
print("\n(IBMQ Backends)")
for backend in provider.backends():
    print(backend.status())

try:
    # select least busy available device and execute.
    least_busy_device = least_busy(provider.backends(simulator=False))
except:
    print("All devices are currently unavailable.")

print("Running on current least busy device: ", least_busy_device)

# Transpile the circuits to make them compatible with the experimental backend
[qc1_new, qc2_new] = transpile(circuits=[qc1, qc2], backend=least_busy_device)

print("Bell circuit before transpile:")
print(qc1)
print("Bell circuit after transpile:")
print(qc1_new)
print("Superposition circuit before transpile:")
print(qc2)
print("Superposition circuit after transpile:")
print(qc2_new)

# Assemble the two circuits into a runnable qobj
qobj = assemble([qc1_new, qc2_new], shots=1000)

# Running qobj on the simulator
sim_job = qasm_simulator.run(qobj)

# Getting the result
sim_result=sim_job.result()

# Show the results
print(sim_result.get_counts(qc1))
print(sim_result.get_counts(qc2))

# Running the job.
exp_job = least_busy_device.run(qobj)

job_monitor(exp_job)
exp_result = exp_job.result()

# Show the results
print(exp_result.get_counts(qc1))
print(exp_result.get_counts(qc2))
