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

"""Example on how to load a file into a QuantumCircuit."""

from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator

circ = QuantumCircuit.from_qasm_file("examples/qasm/entangled_registers.qasm")
print(circ)

# See the backend
sim_backend = BasicSimulator()

# Compile and run the Quantum circuit on a local simulator backend
job_sim = sim_backend.run(circ)
sim_result = job_sim.result()

# Show the results
print("simulation: ", sim_result)
print(sim_result.get_counts(circ))
