# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
PauliTable utility functions.
"""

import numpy as np
from .pauli_table import PauliTable


def pauli_basis(n_qubits, weight=False):
    """Return the ordered PauliTable for the n-qubit Pauli basis.

    Args:
        n_qubits (int): number of qubits
        weight (bool): if True optionally return the basis sorted by Pauli weight
                       rather than lexicographic order (Default: False)

    Returns:
        PauliTable: the PauliTable for the basis
    """
    pauli_1q = PauliTable(np.array([[False, False],
                                    [True, False],
                                    [True, True],
                                    [False, True]],
                                   dtype=np.bool))
    if n_qubits == 1:
        return pauli_1q
    pauli = pauli_1q
    for _ in range(n_qubits - 1):
        pauli = pauli_1q.tensor(pauli)
    if weight:
        return pauli.sort(weight=True)
    return pauli
