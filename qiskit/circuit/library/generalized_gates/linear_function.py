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

from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Clifford
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
        linear: list[list[int]] | np.ndarray | QuantumCircuit,
        validate_input: bool | None = False,
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

        # pylint: disable=cyclic-import
        from qiskit.circuit.library import PermutationGate

        original_circuit = None

        if isinstance(linear, (list, np.ndarray)):
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

        elif isinstance(linear, QuantumCircuit):
            # The following function will raise a CircuitError if there are nonlinear gates.
            original_circuit = linear
            linear = LinearFunction._circuit_to_mat(linear)

        elif isinstance(linear, LinearFunction):
            pass

        elif isinstance(linear, PermutationGate):
            pass

        elif isinstance(linear, Clifford):
            # The following function will raise a CircuitError if clifford does not correspond
            # to a linear function.
            linear = LinearFunction._clifford_to_mat(linear)

        else:
            raise CircuitError(
                "A linear function must be represented either by a list, "
                "a numpy array, or a quantum circuit with linear gates."
            )

        super().__init__(
            name="linear_function", num_qubits=len(linear), params=[linear, original_circuit]
        )

    @staticmethod
    def _circuit_to_mat(qc: QuantumCircuit):
        """This creates a nxn matrix corresponding to the given quantum circuit."""
        nq = qc.num_qubits
        mat = np.eye(nq, nq, dtype=bool)

        for instruction in qc.data:
            if instruction.operation.name in ("barrier", "delay"):
                # can be ignored
                continue
            elif instruction.operation.name == "cx":
                # implemented directly
                cb = qc.find_bit(instruction.qubits[0]).index
                tb = qc.find_bit(instruction.qubits[1]).index
                mat[tb, :] = (mat[tb, :]) ^ (mat[cb, :])
                continue
            elif instruction.operation.name == "swap":
                # implemented directly
                cb = qc.find_bit(instruction.qubits[0]).index
                tb = qc.find_bit(instruction.qubits[1]).index
                mat[[cb, tb]] = mat[[tb, cb]]
                continue

            # In all other cases, we construct the linear function for the operation.
            # and compose (multiply) linear matrices.

            if getattr(instruction.operation, "definition", None) is not None:
                other = LinearFunction(instruction.operation.definition)
            else:
                other = LinearFunction(instruction.operation)

            positions = [qc.find_bit(q).index for q in instruction.qubits]
            other = other.extend_with_identity(len(mat), positions)
            mat = np.dot(other.linear.astype(int), mat.astype(int)) % 2
            mat = mat.astype(bool)

        return mat

    @staticmethod
    def _clifford_to_mat(cliff):
        """This creates a nxn matrix corresponding to the given Clifford, when Clifford
        can be converted to a linear function. This is possible when the clifford has
        tableau of the form [[A, B], [C, D]], with B = C = 0 and D = A^{-1}^t, and zero
        phase vector. In this case, the required matrix is A^t.
        Raises an error otherwise.
        """
        num_qubits = cliff.num_qubits

        if (
            cliff.phase.any()
            or cliff.destab_z.any()
            or cliff.stab_x.any()
            or not np.array_equal(
                np.dot(np.transpose(cliff.destab_x.astype(int)), cliff.stab_z.astype(int)) % 2,
                np.eye(num_qubits),
            )
        ):
            raise CircuitError(
                "The given clifford does not correspond to a linear function."
            )

        return np.transpose(cliff.destab_x)

    def __eq__(self, other):
        """Check if two linear functions represent the same matrix."""
        return (self.linear == other.linear).all()

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

    def extend_with_identity(self, nq: int, positions: list[int]) -> LinearFunction:
        """Extend linear function to a linear function over nq qubits,
        with identities on other subsystems.

        Args:
            nq: number of qubits of the extended function.

            positions: describes the positions of original qubits in the extended
                function's qubits.

        Returns:
            LinearFunction: extended linear function.
        """
        extended_mat = np.eye(nq, dtype=bool)

        for i, pos in enumerate(positions):
            extended_mat[positions, pos] = self.linear[:, i]

        return LinearFunction(extended_mat)
