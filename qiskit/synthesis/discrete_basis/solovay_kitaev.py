# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Synthesize a single qubit gate to a discrete basis set."""

from __future__ import annotations

import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.dagcircuit import DAGCircuit

from .gate_sequence import GateSequence
from .commutator_decompose import commutator_decompose
from .generate_basis_approximations import generate_basic_approximations, _1q_gates, _1q_inverses


class SolovayKitaevDecomposition:
    """The Solovay Kitaev discrete decomposition algorithm.

    This class is called recursively by the transpiler pass, which is why it is separeted.
    See :mod:`qiskit.synthesis.discrete_basis` for more information.
    """

    def __init__(
        self, basic_approximations: str | dict[str, np.ndarray] | list[GateSequence] | None = None
    ) -> None:
        """
        Args:
            basic_approximations: A specification of the basic SU(2) approximations in terms
                of discrete gates. At each iteration this algorithm, the remaining error is
                approximated with the closest sequence of gates in this set.
                If a ``str``, this specifies a ``.npy`` filename from which to load the
                approximation. If a ``dict``, then this contains
                ``{gates: effective_SO3_matrix}`` pairs,
                e.g. ``{"h t": np.array([[0, 0.7071, -0.7071], [0, -0.7071, -0.7071], [-1, 0, 0]]}``.
                If a list, this contains the same information as the dict, but already converted to
                :class:`.GateSequence` objects, which contain the SO(3) matrix and gates.
        """
        if basic_approximations is None:
            # generate a default basic approximation
            basic_approximations = generate_basic_approximations(
                basis_gates=["h", "t", "tdg"], depth=10
            )

        self.basic_approximations = self.load_basic_approximations(basic_approximations)

    def load_basic_approximations(self, data: list | str | dict) -> list[GateSequence]:
        """Load basic approximations.

        Args:
            data: If a string, specifies the path to the file from where to load the data.
                If a dictionary, directly specifies the decompositions as ``{gates: matrix}``.
                There ``gates`` are the names of the gates producing the SO(3) matrix ``matrix``,
                e.g. ``{"h t": np.array([[0, 0.7071, -0.7071], [0, -0.7071, -0.7071], [-1, 0, 0]]}``.

        Returns:
            A list of basic approximations as type ``GateSequence``.

        Raises:
            ValueError: If the number of gate combinations and associated matrices does not match.
        """
        # is already a list of GateSequences
        if isinstance(data, list):
            return data

        # if a file, load the dictionary
        if isinstance(data, str):
            data = np.load(data, allow_pickle=True)

        sequences = []
        for gatestring, matrix in data.items():
            sequence = GateSequence()
            sequence.gates = [_1q_gates[element] for element in gatestring.split()]
            sequence.product = np.asarray(matrix)
            sequences.append(sequence)

        return sequences

    def run(
        self, gate_matrix: np.ndarray, recursion_degree: int, return_dag: bool = False
    ) -> QuantumCircuit | DAGCircuit:
        r"""Run the algorithm.

        Args:
            gate_matrix: The 2x2 matrix representing the gate. Does not need to be SU(2).
            recursion_degree: The recursion degree, called :math:`n` in the paper.
            return_dag: If ``True`` return a :class:`.DAGCircuit`, else a :class:`.QuantumCircuit`.

        Returns:
            A one-qubit circuit approximating the ``gate_matrix`` in the specified discrete basis.
        """
        # make input matrix SU(2) and get the according global phase
        z = 1 / np.sqrt(np.linalg.det(gate_matrix))
        gate_matrix_su2 = GateSequence.from_matrix(z * gate_matrix)
        global_phase = np.arctan2(np.imag(z), np.real(z))

        # get the decompositon as GateSequence type
        decomposition = self._recurse(gate_matrix_su2, recursion_degree)

        # simplify
        _remove_identities(decomposition)
        _remove_inverse_follows_gate(decomposition)

        # convert to a circuit and attach the right phases
        if return_dag:
            out = decomposition.to_dag()
        else:
            out = decomposition.to_circuit()

        out.global_phase = decomposition.global_phase - global_phase

        return out

    def _recurse(self, sequence: GateSequence, n: int) -> GateSequence:
        """Performs ``n`` iterations of the Solovay-Kitaev algorithm on ``sequence``.

        Args:
            sequence: GateSequence to which the Solovay-Kitaev algorithm is applied.
            n: number of iterations that the algorithm needs to run.

        Returns:
            GateSequence that approximates ``sequence``.

        Raises:
            ValueError: if ``u`` does not represent an SO(3)-matrix.
        """
        if sequence.product.shape != (3, 3):
            raise ValueError("Shape of U must be (3, 3) but is", sequence.shape)

        if n == 0:
            return self.find_basic_approximation(sequence)

        u_n1 = self._recurse(sequence, n - 1)

        v_n, w_n = commutator_decompose(sequence.dot(u_n1.adjoint()).product)

        v_n1 = self._recurse(v_n, n - 1)
        w_n1 = self._recurse(w_n, n - 1)
        return v_n1.dot(w_n1).dot(v_n1.adjoint()).dot(w_n1.adjoint()).dot(u_n1)

    def find_basic_approximation(self, sequence: GateSequence) -> Gate:
        """Finds gate in ``self._basic_approximations`` that best represents ``sequence``.

        Args:
            sequence: The gate to find the approximation to.

        Returns:
            Gate in basic approximations that is closest to ``sequence``.
        """
        # TODO explore using a k-d tree here

        def key(x):
            return np.linalg.norm(np.subtract(x.product, sequence.product))

        best = min(self.basic_approximations, key=key)
        return best


def _remove_inverse_follows_gate(sequence):
    index = 0
    while index < len(sequence.gates) - 1:
        curr_gate = sequence.gates[index]
        next_gate = sequence.gates[index + 1]
        if curr_gate.name in _1q_inverses.keys():
            remove = _1q_inverses[curr_gate.name] == next_gate.name
        else:
            remove = curr_gate.inverse() == next_gate

        if remove:
            # remove gates at index and index + 1
            sequence.remove_cancelling_pair([index, index + 1])
            # take a step back to see if we have uncovered a new pair, e.g.
            # [h, s, sdg, h] at index = 1 removes s, sdg but if we continue at index 1
            # we miss the uncovered [h, h] pair at indices 0 and 1
            if index > 0:
                index -= 1
        else:
            # next index
            index += 1


def _remove_identities(sequence):
    index = 0
    while index < len(sequence.gates):
        if sequence.gates[index].name == "id":
            sequence.gates.pop(index)
        else:
            index += 1
