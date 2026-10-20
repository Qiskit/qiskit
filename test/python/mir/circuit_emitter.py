# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit, Qubit, Clbit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.pauli_product_measurement import PauliProductMeasurement
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli


class CircuitEmitter(ABC):
    @abstractmethod
    def new_circuit(self, name):
        pass

    @abstractmethod
    def add_qubit(self, name=None):
        pass

    @abstractmethod
    def add_clbit(self, name=None):
        pass

    @abstractmethod
    def pauli_rotation(self, qubits_indics, pauli_label, angle):
        pass

    @abstractmethod
    def pauli_measurment(self, qubits_indics, pauli_label, clbit_index):
        pass

    @abstractmethod
    def create_circuit(self):
        pass


class InMemoryCircuitEmitter(CircuitEmitter):
    def __init__(self):
        super().__init__()
        self._circuit = None
        self._qubits = []
        self._clbits = []

    def new_circuit(self, name):
        self._circuit = QuantumCircuit(name=name)

    def add_qubit(self, name=None):
        qubit = Qubit()
        self._qubits.append(qubit)
        if name:
            self._circuit.add_register(QuantumRegister(bits=[qubit], name=name))
        else:
            self._circuit.add_bits([qubit])
        return len(self._qubits) - 1  # return the new qubit index

    def add_clbit(self, name=None):
        clbit = Clbit()
        self._clbits.append(clbit)
        if name:
            self._circuit.add_register(ClassicalRegister(bits=[clbit], name=name))
        else:
            self._circuit.add_bits([clbit])
        return len(self._clbits) - 1  # return the new qubit index

    def pauli_rotation(self, qubits_indics, pauli_label, angle):
        # Qiskit convention is little endian (most right char is the lowest bit index)
        # ftir act from left to right [reading_ftir_ir.md "Anatomy of an operation": "pbc_pauli_bases[i], acts on operand i"
        # therefor we reverse the order.
        pauli = Pauli("".join(reversed(pauli_label)))
        # ftir: pbc.putli_rotation is exp(i\ltheta/2*P) [reading_ftir_ir.md op reference table]
        # PauliEvolutionGate docstring:  U(t) = e^{-itH}. so t = -theta/2
        gate = PauliEvolutionGate(pauli, time=-angle / 2)
        self._circuit.append(gate, [self._qubits[qubit_index] for qubit_index in qubits_indics])

    def pauli_measurment(self, qubits_indics, pauli_label, clbit_index):
        # Qiskit convention is little endian (most right char is the lowest bit index)
        # ftir act from left to right [reading_ftir_ir.md "Anatomy of an operation": "pbc_pauli_bases[i], acts on operand i"
        # therefor we reverse the order.
        pauli = Pauli("".join(reversed(pauli_label)))
        self._circuit.append(
            PauliProductMeasurement(pauli),
            [self._qubits[qubit_index] for qubit_index in qubits_indics],
            [self._clbits[clbit_index]],
        )

    def create_circuit(self):
        return self._circuit


class PythonFileCircuitEmitter(CircuitEmitter):
    _PREAMBLE = (
        "from qiskit.circuit import QuantumCircuit, Qubit, Clbit, QuantumRegister, ClassicalRegister\n"
        "from qiskit.circuit.library.pauli_product_measurement import PauliProductMeasurement\n"
        "from qiskit.circuit.library import PauliEvolutionGate\n"
        "from qiskit.quantum_info import Pauli\n"
    )

    def __init__(self, target_path):
        super().__init__()
        self._target_path = target_path
        self._lines = []
        self._qubits_counter = -1
        self._clbits_counter = -1
        self._lines.append("qubits = []")
        self._lines.append("clbits = []")

    def new_circuit(self, name):
        self._lines.append(f"circuit = QuantumCircuit(name={name!r})")

    def add_qubit(self, name=None):
        self._lines.append("qubits.append(Qubit())")
        if name:
            self._lines.append(
                f"circuit.add_register(QuantumRegister(bits=[qubits[-1]], name={name!r}))"
            )
        else:
            self._lines.append("circuit.add_bits([qubits[-1]])")
        self._qubits_counter += 1
        return self._qubits_counter

    def add_clbit(self, name=None):
        self._lines.append("clbits.append(Clbit())")
        if name:
            self._lines.append(
                f"circuit.add_register(ClassicalRegister(bits=[clbits[-1]], name={name!r}))"
            )
        else:
            self._lines.append("circuit.add_bits([clbits[-1]])")
        self._clbits_counter += 1
        return self._clbits_counter

    def pauli_rotation(self, qubits_indics, pauli_label, angle):
        # Qiskit convention is little endian (most right char is the lowest bit index)
        # ftir act from left to right [reading_ftir_ir.md "Anatomy of an operation": "pbc_pauli_bases[i], acts on operand i"
        # therefor we reverse the order.
        self._lines.append(f'pauli = Pauli("".join(reversed({pauli_label!r})))')
        # ftir: pbc.putli_rotation is exp(i\ltheta/2*P) [reading_ftir_ir.md op reference table]
        # PauliEvolutionGate docstring:  U(t) = e^{-itH}. so t = -theta/2
        self._lines.append(f"gate = PauliEvolutionGate(pauli, time=-{angle!r}/2)")
        self._lines.append(
            f"circuit.append(gate, [qubits[qubit_index] for qubit_index in {qubits_indics}])"
        )

    def pauli_measurment(self, qubits_indics, pauli_label, clbit_index):
        # Qiskit convention is little endian (most right char is the lowest bit index)
        # ftir act from left to right [reading_ftir_ir.md "Anatomy of an operation": "pbc_pauli_bases[i], acts on operand i"
        # therefor we reverse the order.
        self._lines.append(f'pauli = Pauli("".join(reversed({pauli_label!r})))')
        self._lines.append(
            f"circuit.append(PauliProductMeasurement(pauli), [qubits[qubit_index] for qubit_index in {qubits_indics!r}], [clbits[{clbit_index!r}]])"
        )

    def create_circuit(self):
        with open(self._target_path, "w") as file:
            file.write(self._PREAMBLE)
            file.write("\n")
            file.write("\n".join(self._lines))
            file.write("\n")
            file.write("print(circuit)")
            file.write("\n")
        return "Done"
