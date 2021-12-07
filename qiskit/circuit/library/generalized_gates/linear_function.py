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
from qiskit.circuit.quantumcircuit import QuantumCircuit, Instruction
from qiskit.circuit.exceptions import CircuitError
import numpy as np


class LinearFunction(QuantumCircuit):
    def __init__(
            self,
            num_qubits: int,
            linear: Union[List[List[int]], np.ndarray, QuantumCircuit]):

        """Create a new linear function gate."""

        if isinstance(linear, QuantumCircuit):
            mat = _linear_quantum_circuit_to_mat(linear)
        else:
            mat = linear

        from qiskit.transpiler.synthesis import cnot_synth
        circuit = cnot_synth(mat)

        super().__init__(num_qubits, name="LinearFunction")
        self.append(circuit, self.qubits)


def _linear_quantum_circuit_to_mat(qc: QuantumCircuit):
    nq = qc.num_qubits
    mat = np.eye(nq, nq, dtype=bool)
    bit_indices = {bit: idx for idx, bit in enumerate(qc.qubits)}

    for inst, qargs, cargs in qc.data:
        if inst.name == "cx":
            c = bit_indices[qargs[0]]
            t = bit_indices[qargs[1]]
            mat[t, :] = (mat[t, :]) ^ (mat[c, :])
        elif inst.name == "swap":
            c = bit_indices[qargs[0]]
            t = bit_indices[qargs[1]]
            mat[[c, t]] = mat[[t, c]]
        else:
            raise CircuitError(
                "A linear quantum circuit can include only CX and SWAP gates."
            )

    return mat
