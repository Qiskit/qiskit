from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import random_clifford


class BasicSimulatorGHZBenchmark:
    """Benchmark BasicSimulator on GHZ Clifford circuits."""

    params = ([4, 8, 12, 16],)
    param_names = ("n_qubits",)

    def setup(self, n_qubits):
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n_qubits), range(n_qubits))
        self.ghz_circuit = qc

    def time_statevector(self, n_qubits):
        backend = BasicSimulator()
        backend.run(
            self.ghz_circuit,
            shots=1024,
            use_clifford_optimization=False,
        ).result()

    def time_clifford(self, n_qubits):
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
        cliff = random_clifford(n_qubits)
        qc = cliff.to_circuit()
        qc.measure_all()
        self.clifford_circuit = qc

    def time_statevector(self, n_qubits):
        backend = BasicSimulator()
        backend.run(
            self.clifford_circuit,
            shots=1024,
            use_clifford_optimization=False,
        ).result()

    def time_clifford(self, n_qubits):
        backend = BasicSimulator()
        backend.run(
            self.clifford_circuit,
            shots=1024,
            use_clifford_optimization=True,
        ).result()
