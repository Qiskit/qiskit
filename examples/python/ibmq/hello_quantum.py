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

"""Example used in the README. In this example a Bell state is made."""

# Import Qiskit
from qiskit import QuantumCircuit
from qiskit import execute, IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy

# Create a Quantum Circuit
qc = QuantumCircuit(2, 2)

# Add a H gate on qubit 0, putting this qubit in superposition.
qc.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
qc.cx(0, 1)
# Add a Measure gate to see the state.
qc.measure([0, 1], [0, 1])

# See a list of available local simulators
print("BasicAer backends: ", BasicAer.backends())
backend_sim = BasicAer.get_backend('qasm_simulator')

# Compile and run the Quantum circuit on a simulator backend
job_sim = execute(qc, backend_sim)
result_sim = job_sim.result()

# Show the results
print(result_sim.get_counts(qc))

# Authenticate for access to remote backends
try:
    provider = IBMQ.load_account()
except:
    print("""WARNING: No valid IBMQ credentials found on disk.
             You must store your credentials using IBMQ.save_account(token, url).
             For now, there's only access to local simulator backends...""")
    exit(0)

# see a list of available remote backends
ibmq_backends = provider.backends()

print("Remote backends: ", ibmq_backends)
# Compile and run the Quantum Program on a real device backend
# select those with at least 2 qubits
try:
    least_busy_device = least_busy(provider.backends(
        filters=lambda x: x.configuration().n_qubits >= 2, simulator=False))
except:
    print("All devices are currently unavailable.")

print("Running on current least busy device: ", least_busy_device)

#running the job
job_exp = execute(qc, least_busy_device, shots=1024, max_credits=10)
result_exp = job_exp.result()

# Show the results
print('Counts: ', result_exp.get_counts(qc))
