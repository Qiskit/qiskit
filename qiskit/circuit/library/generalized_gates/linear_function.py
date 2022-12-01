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
from qiskit.synthesis.linear import check_invertible_binary_matrix


class LinearFunction(Gate):
    r"""A linear reversible circuit on n qubits.

    Internally, a linear function acting on n qubits is represented
    as a n x n matrix of 0s and 1s in numpy array format.

    A linear function can be synthesized into CX and SWAP gates using the Patel–Markov–Hayes
    algorithm, as implemented in :func:`~qiskit.transpiler.synthesis.cnot_synth`
    based on reference [1].

    For efficiency, the internal n x n matrix is stored in the format expected
    by cnot_synth, which is the big-endian (and not the little-endian) bit-ordering convention.

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


    **References:**

    [1] Ketan N. Patel, Igor L. Markov, and John P. Hayes,
    Optimal synthesis of linear reversible circuits,
    Quantum Inf. Comput. 8(3) (2008).
    `Online at umich.edu. <https://web.eecs.umich.edu/~imarkov/pubs/jour/qic08-cnot.pdf>`_
    """

    def __init__(
        self,
        linear: Union[List[List[int]], np.ndarray, QuantumCircuit],
        validate_input: Optional[bool] = False,
    ) -> None:
        """Create a new linear function.

        Args:
            linear (list[list] or ndarray[bool] or QuantumCircuit):
                either an n x n matrix, describing the linear function,
                or a quantum circuit composed of linear gates only
                (currently supported gates are CX and SWAP).

            validate_input: if True, performs more expensive input validation checks,
                such as checking that a given n x n matrix is invertible.

        Raises:
            CircuitError: if the input is invalid:
                either a matrix is non {square, invertible},
                or a quantum circuit contains non-linear gates.

        """
        if not isinstance(linear, (list, np.ndarray, QuantumCircuit)):
            raise CircuitError(
                "A linear function must be represented either by a list, "
                "a numpy array, or a quantum circuit with linear gates."
            )

        if isinstance(linear, QuantumCircuit):
            # The following function will raise a CircuitError if there are nonlinear gates.
            original_circuit = linear
            linear = _linear_quantum_circuit_to_mat(linear)

        else:
            original_circuit = None

            # Normalize to numpy array (coercing entries to 0s and 1s)
            try:
                linear = np.array(linear, dtype=bool, copy=True)
            except ValueError:
                raise CircuitError(
                    "A linear function must be represented by a square matrix."
                ) from None

            # Check that the matrix is square
            if len(linear.shape) != 2 or linear.shape[0] != linear.shape[1]:
                raise CircuitError("A linear function must be represented by a square matrix.")

            # Optionally, check that the matrix is invertible
            if validate_input:
                if not check_invertible_binary_matrix(linear):
                    raise CircuitError(
                        "A linear function must be represented by an invertible matrix."
                    )

        super().__init__(
            name="linear_function", num_qubits=len(linear), params=[linear, original_circuit]
        )

    def validate_parameter(self, parameter):
        """Parameter validation"""
        return parameter

    def _define(self):
        """Populates self.definition with a decomposition of this gate."""
        self.definition = self.synthesize()

    def synthesize(self):
        """Synthesizes the linear function into a quantum circuit.

        Returns:
            QuantumCircuit: A circuit implementing the evolution.
        """
        from qiskit.synthesis.linear import synth_cnot_count_full_pmh

        return synth_cnot_count_full_pmh(self.linear)

    @property
    def linear(self):
        """Returns the n x n matrix representing this linear function"""
        return self.params[0]

    @property
    def original_circuit(self):
        """Returns the original circuit used to construct this linear function
        (including None, when the linear function is not constructed from a circuit).
        """
        return self.params[1]

    def is_permutation(self) -> bool:
        """Returns whether this linear function is a permutation,
        that is whether every row and every column of the n x n matrix
        has exactly one 1.
        """
        linear = self.linear
        perm = np.all(np.sum(linear, axis=0) == 1) and np.all(np.sum(linear, axis=1) == 1)
        return perm

    def permutation_pattern(self):
        """This method first checks if a linear function is a permutation and raises a
        `qiskit.circuit.exceptions.CircuitError` if not. In the case that this linear function
        is a permutation, returns the permutation pattern.
        """
        if not self.is_permutation():
            raise CircuitError("The linear function is not a permutation")

        linear = self.linear
        locs = np.where(linear == 1)
        return locs[1]


def _linear_quantum_circuit_to_mat(qc: QuantumCircuit):
    """This creates a n x n matrix corresponding to the given linear quantum circuit."""
    nq = qc.num_qubits
    mat = np.eye(nq, nq, dtype=bool)

    for instruction in qc.data:
        if instruction.operation.name == "cx":
            cb = qc.find_bit(instruction.qubits[0]).index
            tb = qc.find_bit(instruction.qubits[1]).index
            mat[tb, :] = (mat[tb, :]) ^ (mat[cb, :])
        elif instruction.operation.name == "swap":
            cb = qc.find_bit(instruction.qubits[0]).index
            tb = qc.find_bit(instruction.qubits[1]).index
            mat[[cb, tb]] = mat[[tb, cb]]
        else:
            raise CircuitError("A linear quantum circuit can include only CX and SWAP gates.")

    return mat
