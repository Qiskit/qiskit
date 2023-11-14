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


def synth_qft_line(num_qubits, do_swaps=True):
    """Synthesis of a QFT circuit for a linear nearest neighbor connectivity.
    Based on Fowler et al. Fig 2.b from https://arxiv.org/abs/quant-ph/0402196"""

    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.h(num_qubits - 1)
        for j in range(i, num_qubits - 1):
            qc.p(np.pi / 2 ** (j - i + 2), num_qubits - j + i - 1)
            qc.cx(num_qubits - j + i - 1, num_qubits - j + i - 2)
            qc.p(-np.pi / 2 ** (j - i + 2), num_qubits - j + i - 2)
            qc.cx(num_qubits - j + i - 2, num_qubits - j + i - 1)
            qc.cx(num_qubits - j + i - 1, num_qubits - j + i - 2)
            qc.p(np.pi / 2 ** (j - i + 2), num_qubits - j + i - 1)

    if not do_swaps:
        # Add a reversal network for LNN connectivity in depth 2*n+2,
        # based on Kutin at al., https://arxiv.org/abs/quant-ph/0701194, Section 5
        for _ in range((num_qubits + 1) // 2):
            qc = _append_cx_stage1(qc, num_qubits)
            qc = _append_cx_stage2(qc, num_qubits)
        if (num_qubits % 2) == 0:
            qc = _append_cx_stage1(qc, num_qubits)

    return qc
