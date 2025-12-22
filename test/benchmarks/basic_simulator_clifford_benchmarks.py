# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#

"""BasicSimulator Clifford benchmarks."""

from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import random_clifford

# pylint: disable=attribute-defined-outside-init


class BasicSimulatorGHZBenchmark:
    """Benchmark BasicSimulator on GHZ Clifford circuits."""

    params = ([4, 8, 12, 16],)
    param_names = ("n_qubits",)

    def setup(self, n_qubits):
        """Setup GHZ circuit for given n_qubits."""

        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n_qubits), range(n_qubits))
        self.ghz_circuit = qc

    def time_statevector(self, n_qubits):
        """Time statevector simulation of GHZ circuit."""
        _ = n_qubits  # ASV parameter
        backend = BasicSimulator()
        backend.run(
            self.ghz_circuit,
            shots=1024,
            use_clifford_optimization=False,
        ).result()

    def time_clifford(self, n_qubits):
        """Time Clifford-optimized simulation of GHZ circuit."""
        _ = n_qubits  # ASV parameter
        backend = BasicSimulator()
        backend.run(
            self.ghz_circuit,
            shots=1024,
            use_clifford_optimization=True,
        ).result()


class BasicSimulatorRandomCliffordBenchmark:
    """Benchmark BasicSimulator on random Clifford circuits."""

    params = ([4, 8, 12, 16],)
    param_names = ("n_qubits",)

    def setup(self, n_qubits):
        """Setup random Clifford circuit for given n_qubits."""

        cliff = random_clifford(n_qubits, seed=0)
        qc = cliff.to_circuit()
        qc.measure_all()
        self.clifford_circuit = qc

    def time_statevector(self, n_qubits):
        """Time statevector simulation of random Clifford circuit."""
        _ = n_qubits  # ASV parameter
        backend = BasicSimulator()
        backend.run(
            self.clifford_circuit,
            shots=1024,
            use_clifford_optimization=False,
        ).result()

    def time_clifford(self, n_qubits):
        """Time Clifford-optimized simulation of random Clifford circuit."""
        _ = n_qubits  # ASV parameter
        backend = BasicSimulator()
        backend.run(
            self.clifford_circuit,
            shots=1024,
            use_clifford_optimization=True,
        ).result()
