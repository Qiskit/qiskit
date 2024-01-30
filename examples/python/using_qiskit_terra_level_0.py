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
Example showing how to use Qiskit at introduction level.

This example shows the most basic way to use Qiskit. It builds some circuits
and runs them on both the BasicAer (local Qiskit provider) or IBM Quantum (remote IBM Quantum provider).

To control the compile parameters we have provided a transpile function which can be used
as a level 1 user.

"""

# Import the Qiskit modules
from qiskit import QuantumCircuit
from qiskit import transpile, BasicAer

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
print("(BasicAER Backends)")
print(BasicAer.backends())

# running the job
sim_backend = BasicAer.get_backend("qasm_simulator")
job_sim = sim_backend.run(transpile([qc1, qc2], sim_backend))
sim_result = job_sim.result()

# Show the results
print(sim_result.get_counts(qc1))
print(sim_result.get_counts(qc2))
