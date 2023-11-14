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


def synth_qft_line(num_qubits):
    """Synthesis of a QFT circuit for a linear nearest neighbor connectivity."""

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

    return qc
