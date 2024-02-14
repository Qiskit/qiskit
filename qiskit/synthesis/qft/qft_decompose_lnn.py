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
Circuit synthesis for a QFT circuit.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear_phase.cz_depth_lnn import _append_cx_stage1, _append_cx_stage2


def synth_qft_line(
    num_qubits: int, do_swaps: bool = True, approximation_degree: int = 0
) -> QuantumCircuit:
    """Synthesis of a QFT circuit for a linear nearest neighbor connectivity.
    Based on Fig 2.b in Fowler et al. [1].

    Note that this method *reverts* the order of qubits in the circuit,
    compared to the original :class:`.QFT` code.
    Hence, the default value of the ``do_swaps`` parameter is ``True``
    since it produces a circuit with fewer CX gates.

    Args:
        num_qubits: The number of qubits on which the QFT acts.
        approximation_degree: The degree of approximation (0 for no approximation).
        do_swaps: Whether to include the final swaps in the QFT.

    Returns:
        A circuit implementation of the QFT circuit.

    References:
        1. A. G. Fowler, S. J. Devitt, and L. C. L. Hollenberg,
           *Implementation of Shor's algorithm on a linear nearest neighbour qubit array*,
           Quantum Info. Comput. 4, 4 (July 2004), 237–251.
           `arXiv:quant-ph/0402196 [quant-ph] <https://arxiv.org/abs/quant-ph/0402196>`_
    """

    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.h(num_qubits - 1)

        for j in range(i, num_qubits - 1):
            if j - i + 2 < num_qubits - approximation_degree + 1:
                qc.p(np.pi / 2 ** (j - i + 2), num_qubits - j + i - 1)
                qc.cx(num_qubits - j + i - 1, num_qubits - j + i - 2)
                qc.p(-np.pi / 2 ** (j - i + 2), num_qubits - j + i - 2)
                qc.cx(num_qubits - j + i - 2, num_qubits - j + i - 1)
                qc.cx(num_qubits - j + i - 1, num_qubits - j + i - 2)
                qc.p(np.pi / 2 ** (j - i + 2), num_qubits - j + i - 1)
            else:
                qc.cx(num_qubits - j + i - 1, num_qubits - j + i - 2)
                qc.cx(num_qubits - j + i - 2, num_qubits - j + i - 1)
                qc.cx(num_qubits - j + i - 1, num_qubits - j + i - 2)

    if not do_swaps:
        # Add a reversal network for LNN connectivity in depth 2*n+2,
        # based on Kutin at al., https://arxiv.org/abs/quant-ph/0701194, Section 5.
        for _ in range((num_qubits + 1) // 2):
            qc = _append_cx_stage1(qc, num_qubits)
            qc = _append_cx_stage2(qc, num_qubits)
        if (num_qubits % 2) == 0:
            qc = _append_cx_stage1(qc, num_qubits)

    return qc
