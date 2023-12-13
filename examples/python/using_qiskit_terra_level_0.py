# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
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
and runs them on both the TestProvider(local Qiskit provider).

To control the compile parameters we have provided a transpile function which can be used
as a level 1 user.

"""

# Import the Qiskit modules
from qiskit import QuantumCircuit
from qiskit import execute, TestProvider

# making first circuit: bell state
qc1 = QuantumCircuit(2, 2)
qc1.h(0)
qc1.cx(0, 1)
qc1.measure([0, 1], [0, 1])

# making another circuit: superpositions
qc2 = QuantumCircuit(2, 2)
qc2.h([0, 1])
qc2.measure([0, 1], [0, 1])

# setting up the backend
print("(TestProvider Backends)")
print(TestProvider.backends())

# running the job
job_sim = execute([qc1, qc2], TestProvider.get_backend("test_simulator"))
sim_result = job_sim.result()

# Show the results
print(sim_result.get_counts(qc1))
print(sim_result.get_counts(qc2))
