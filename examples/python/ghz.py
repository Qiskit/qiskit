# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
GHZ state example. It also compares running on experiment and simulator

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, BasicAer, execute
from qiskit.providers.ibmq import least_busy


###############################################################
# Make a quantum circuit for the GHZ state.
###############################################################
q = QuantumRegister(5, "q")
c = ClassicalRegister(5, "c")
qc = QuantumCircuit(q, c, name='ghz')

# Create a GHZ state
qc.h(q[0])
for i in range(4):
    qc.cx(q[i], q[i+1])
# Insert a barrier before measurement
qc.barrier()
# Measure all of the qubits in the standard basis
for i in range(5):
    qc.measure(q[i], c[i])

###############################################################
# Set up the API and execute the program.
###############################################################
try:
    IBMQ.load_accounts()
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a file with your personal token?
             For now, there's only access to local simulator backends...""")

# First version: simulator
sim_backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, sim_backend, shots=1024)
result = job.result()
print('Qasm simulator')
print(result.get_counts(qc))

# Second version: real device
least_busy_device = least_busy(IBMQ.backends(simulator=False,
                                             filters=lambda x: x.configuration().n_qubits > 4))
print("Running on current least busy device: ", least_busy_device)
job = execute(qc, least_busy_device, shots=1024)
result = job.result()
print(result.get_counts(qc))
