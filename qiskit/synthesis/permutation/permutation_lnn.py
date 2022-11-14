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

"""Depth-efficient synthesis algorithm for Permutation gates."""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from .permutation_utils import _inverse_pattern


def synth_permutation_depth_lnn_kms(pattern):
    """Synthesize a permutation circuit for a linear nearest-neighbor
    architecture using the Kutin, Moulton, Smithline method.

    This is the permutation synthesis algorithm from
    https://arxiv.org/abs/quant-ph/0701194, Chapter 6.
    It synthesizes any permutation of n qubits over linear nearest-neighbor
    architecture using SWAP gates with depth at most n and size at most
    n(n-1)/2 (where both depth and size are measured with respect to SWAPs).

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

    # In Qiskit, the permutation pattern [2, 4, 3, 0, 1] means that
    # the permutation that maps qubit 2 to position 0, 4 to 1, 3 to 2, 0 to 3, and 1 to 4.
    # In the permutation synthesis code below the notation is opposite:
    # [2, 4, 3, 0, 1] means that 0 maps to 2, 1 to 3, 2 to 3, 3 to 0, and 4 to 1.
    # This is why we invert the pattern.
    cur_pattern = _inverse_pattern(pattern)

    num_qubits = len(cur_pattern)
    qc = QuantumCircuit(num_qubits)

    # add conditional odd-even swap layers
    for i in range(num_qubits):
        _create_swap_layer(qc, cur_pattern, i % 2)

    return qc


def _create_swap_layer(qc, pattern, starting_point):
    """Implements a single swap layer, consisting of conditional swaps between each
    neighboring couple. The starting_point is the first qubit to use (either 0 or 1
    for even or odd layers respectively). Mutates both the quantum circuit ``qc``
    and the permutation pattern ``pattern``.
    """
    num_qubits = len(pattern)
    for j in range(starting_point, num_qubits - 1, 2):
        if pattern[j] > pattern[j + 1]:
            qc.swap(j, j + 1)
            pattern[j], pattern[j + 1] = pattern[j + 1], pattern[j]
