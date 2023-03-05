# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for a stabilizer state preparation circuit.
"""
# pylint: disable=invalid-name

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states import StabilizerState
from qiskit.synthesis.clifford.clifford_decompose_layers import synth_clifford_layers


def synth_stabilizer_layers(stab):
    """Synthesis of a stabilizer state into layers."""

    if not isinstance(stab, StabilizerState):
        raise QiskitError("The input is not a StabilizerState.")

    cliff = stab.clifford
    num_qubits = cliff.num_qubits
    qubit_list = list(range(num_qubits))

    circ = synth_clifford_layers(cliff)
    H2_circ = circ.data[3]
    S1_circ = circ.data[4]
    CZ1_circ = circ.data[5]
    H1_circ = circ.data[6]
    pauli_circ = circ.data[7]

    stab_circuit = QuantumCircuit(num_qubits)
    stab_circuit.append(H2_circ, qubit_list)
    stab_circuit.append(S1_circ, qubit_list)
    stab_circuit.append(CZ1_circ, qubit_list)
    stab_circuit.append(H1_circ, qubit_list)
    stab_circuit.append(pauli_circ, qubit_list)

    return stab_circuit
