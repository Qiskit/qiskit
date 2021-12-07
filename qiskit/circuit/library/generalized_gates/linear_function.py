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

"""Linear Function."""

from typing import Union, List
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


class LinearFunction(QuantumCircuit):
    """An n_qubit linear circuit."""

    def __init__(
        self, num_qubits: int, linear: Union[List[List[int]], np.ndarray, QuantumCircuit]
    ) -> None:
        """Return a linear circuit implemented using SWAPs and CXs.

        Represents a linear function acting on n qubits as a n x n matrix.

        Args:
            num_qubits (int): circuit width.
            linear (list[list] or ndarray or QuantumCircuit):
                n x n matrix, describing the state of the input circuit, or a quantum circuit
                composed of linear gates only

        Raises:
            QiskitError: if the input quantum circuit contains non-linear gates.
        """
        if isinstance(linear, QuantumCircuit):
            mat = _linear_quantum_circuit_to_mat(linear)
        else:
            mat = linear

        from qiskit.transpiler.synthesis import cnot_synth

        circuit = cnot_synth(mat)

        super().__init__(num_qubits, name="LinearFunction")
        self.append(circuit, self.qubits)


def _linear_quantum_circuit_to_mat(qc: QuantumCircuit):
    """This creates a n x n matrix corresponding to the given linear quantum circuit."""
    nq = qc.num_qubits
    mat = np.eye(nq, nq, dtype=bool)
    bit_indices = {bit: idx for idx, bit in enumerate(qc.qubits)}

    for inst, qargs, _ in qc.data:
        if inst.name == "cx":
            cb = bit_indices[qargs[0]]
            tb = bit_indices[qargs[1]]
            mat[tb, :] = (mat[tb, :]) ^ (mat[cb, :])
        elif inst.name == "swap":
            cb = bit_indices[qargs[0]]
            tb = bit_indices[qargs[1]]
            mat[[cb, tb]] = mat[[tb, cb]]
        else:
            raise CircuitError("A linear quantum circuit can include only CX and SWAP gates.")

    return mat
