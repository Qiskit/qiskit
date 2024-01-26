# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
GHZ state example. It also compares running on experiment and simulator.
"""

from qiskit import QuantumCircuit
from qiskit import BasicAer, transpile


###############################################################
# Make a quantum circuit for the GHZ state.
###############################################################
num_qubits = 5
qc = QuantumCircuit(num_qubits, num_qubits, name="ghz")

# Create a GHZ state
qc.h(0)
for i in range(num_qubits - 1):
    qc.cx(i, i + 1)
# Insert a barrier before measurement
qc.barrier()
# Measure all of the qubits in the standard basis
for i in range(num_qubits):
    qc.measure(i, i)

sim_backend = BasicAer.get_backend("qasm_simulator")
job = sim_backend.run(transpile(qc, sim_backend), shots=1024)
result = job.result()
print("Qasm simulator : ")
print(result.get_counts(qc))
