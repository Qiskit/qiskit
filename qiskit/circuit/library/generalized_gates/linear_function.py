# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear Function."""

from typing import Union, List, Optional
import numpy as np
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError


class LinearFunction(Gate):
    """An n_qubit linear circuit."""

    def __init__(
        self,
        linear: Union[List[List[int]], np.ndarray, QuantumCircuit],
        validate_input: Optional[bool] = False,
    ) -> None:
        """Create a new linear function.

        Internally, represents a linear function acting on n qubits as a n x n matrix of 0s and 1s
        in numpy array format.

        Args:
            linear (list[list] or ndarray or QuantumCircuit):
                either an n x n matrix, describing the linear function,
                or a quantum circuit composed of linear gates only
                (currently supported gates are CX and SWAP).

            validate_input: if True, performs more expensive input validation checks,
                such as checking that a given n x n matrix is invertible.

        Raises:
            CircuitError: if the input is invalid:
                either a matrix is non {square, invertible},
                or a quantum circuit contains non-linear gates.

        **Example:** the circuit

        .. parsed-literal::

            q_0: ──■──
                 ┌─┴─┐
            q_1: ┤ X ├
                 └───┘
            q_2: ─────

        is represented by a 3x3 linear matrix

        .. math::

                \begin{pmatrix}
                    1 & 0 & 0 \\
                    1 & 1 & 0 \\
                    0 & 0 & 1
                \end{pmatrix}

        """
        if not isinstance(linear, (list, np.ndarray, QuantumCircuit)):
            raise CircuitError(
                "A linear function must be represented either by a list, "
                "a numpy array, or a quantum circuit with linear gates."
            )

        if isinstance(linear, QuantumCircuit):
            # The following function will raise a CircuitError if there are nonlinear gates.
            self._mat = _linear_quantum_circuit_to_mat(linear)

        else:
            # Normalize to numpy array (coercing entries to 0s and 1s)
            ok = True
            try:
                self._mat = np.array(linear, dtype=bool, copy=False)
            except ValueError:
                ok = False

            if not ok:
                raise CircuitError("A linear function must be represented by a square matrix.")

            # Check that the matrix is square
            if not len(self._mat.shape) == 2 or self._mat.shape[0] != self._mat.shape[1]:
                raise CircuitError("A linear function must be represented by a square matrix.")

            # Optionally, check that the matrix is invertible
            if validate_input:
                det = np.linalg.det(linear) % 2
                if not np.allclose(det, 1):
                    raise CircuitError(
                        "A linear function must be represented by an invertible matrix."
                    )

        super().__init__(name="linear_function", num_qubits=len(self._mat), params=[])

    def _define(self):
        """Populates self.definition with a decomposition of this gate."""
        self.definition = self.synthesize()

    def synthesize(self):
        """Synthesize ``qiskit.circuit.library.LinearFunction``.

        Returns:
            QuantumCircuit: A circuit implementing the evolution.
        """
        from qiskit.transpiler.synthesis import cnot_synth

        return cnot_synth(self._mat)

    @property
    def matrix(self):
        """Returns the n x n matrix representing this linear function"""
        return self._mat


def _linear_quantum_circuit_to_mat(qc: QuantumCircuit):
    """This creates a n x n matrix corresponding to the given linear quantum circuit."""
    nq = qc.num_qubits
    mat = np.eye(nq, nq, dtype=bool)

    for inst, qargs, _ in qc.data:
        if inst.name == "cx":
            cb = qc.find_bit(qargs[0]).index
            tb = qc.find_bit(qargs[1]).index
            mat[tb, :] = (mat[tb, :]) ^ (mat[cb, :])
        elif inst.name == "swap":
            cb = qc.find_bit(qargs[0]).index
            tb = qc.find_bit(qargs[1]).index
            mat[[cb, tb]] = mat[[tb, cb]]
        else:
            raise CircuitError("A linear quantum circuit can include only CX and SWAP gates.")

    return mat
