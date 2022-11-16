# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Synthesis algorithm for Permutation gates for full-connectivity."""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from .permutation_utils import _get_ordered_swap


def synth_permutation_basic(pattern):
    """Synthesize a permutation circuit for a fully-connected
    architecture using sorting.

    More precisely, if the input permutation is a cycle of length ``m``,
    then this creates a quantum circuit with ``m-1`` SWAPs (and of depth ``m-1``);
    if the input  permutation consists of several disjoint cycles, then each cycle
    is essentially treated independently.

    Args:
        pattern (Union[list[int], np.ndarray]): permutation pattern, describing
            which qubits occupy the positions 0, 1, 2, etc. after applying the
            permutation. That is, ``pattern[k] = m`` when the permutation maps
            qubit ``m`` to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``
            means that qubit ``2`` goes to position ``0``, qubit ``4`` goes to
            position ``1``, etc.

    Returns:
        QuantumCircuit: the synthesized quantum circuit.
    """
    # This is the very original Qiskit algorithm for synthesizing permutations.

    num_qubits = len(pattern)
    qc = QuantumCircuit(num_qubits)

    swaps = _get_ordered_swap(pattern)

    for swap in swaps:
        qc.swap(swap[0], swap[1])

    return qc
