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

# Import the Qiskit modules
from qiskit import IBMQ, BasicAer
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile, assemble
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor

from qiskit.transpiler import CouplingMap


provider = IBMQ.load_account()

# Making first circuit: superpositions
qc1 = QuantumCircuit(4, 4)
qc1.h(0)
qc1.cx(0, 1)
qc1.measure([0, 1], [0, 1])

# Making another circuit: GHZ State
qc2 = QuantumCircuit(4, 4)
qc2.h([0])
qc2.cx(0, 1)
qc2.cx(0, 2)
qc2.cx(0, 3)
qc2.measure([0, 1, 2, 3], [0, 1, 2, 3])


def status_header(sort):
    print(f"{sort:<25} |  jobs | status")
    print("===========================================")


def format_status(status):
    status = status.to_dict()
    print(f"{status['backend_name'][:25]:>25} | {status['pending_jobs']:5d} | {status['status_msg'][:6]:>6}")


# Setting up the backend
status_header("Aer Backends")
for backend in BasicAer.backends():
    format_status(backend.status())
qasm_simulator = BasicAer.get_backend('qasm_simulator')


# Compile and run the circuit on a real device backend
# See a list of available remote backends
status_header("IBMQ Backends")
for backend in provider.backends():
    format_status(backend.status())

try:
    # select least busy available device and execute.
    least_busy_device = least_busy([backend for backend in provider.backends(simulator=False)
                                    if backend.status().pending_jobs > 0])
except:
    print("All devices are currently unavailable.")

print("Running on current least busy device: ", least_busy_device)


# compiling the circuits
coupling_map = CouplingMap(least_busy_device.configuration().coupling_map)
print("coupling map: ", coupling_map)

qc1_new = transpile(qc1, backend=least_busy_device)
qc2_new = transpile(qc2, backend=least_busy_device)

print("Bell circuit before passes:")
print(qc1)
print("Bell circuit after passes:")
print(qc1_new)
print("Superposition circuit before passes:")
print(qc2)
print("Superposition circuit after passes:")
print(qc2_new)

# Assemble the two circuits into a runnable qobj
qobj = assemble([qc1_new, qc2_new], shots=1000)

# Running qobj on the simulator
print("Running on simulator:")
sim_job = qasm_simulator.run(qobj)

# Getting the result
sim_result = sim_job.result()

# Show the results
print(sim_result.get_counts(qc1))
print(sim_result.get_counts(qc2))

# Running the job.
print("Running on device:")
exp_job = least_busy_device.run(qobj)

job_monitor(exp_job)
exp_result = exp_job.result()

# Show the results
print(exp_result.get_counts(qc1))
print(exp_result.get_counts(qc2))
