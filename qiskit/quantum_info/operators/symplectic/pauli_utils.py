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
PauliList utility functions.
"""

import warnings

import numpy as np

from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit.quantum_info.operators.symplectic.pauli_table import PauliTable


def pauli_basis(num_qubits, weight=False, pauli_list=False):
    """Return the ordered PauliTable or PauliList for the n-qubit Pauli basis.

    Args:
        num_qubits (int): number of qubits
        weight (bool): if True optionally return the basis sorted by Pauli weight
                       rather than lexicographic order (Default: False)
        pauli_list (bool): if True, the return type becomes PauliList, otherwise PauliTable.

    Returns:
        PauliTable, PauliList: the Paulis for the basis
    """
    if pauli_list:
        pauli_1q = PauliList(["I", "X", "Y", "Z"])
    else:
        warnings.warn(
            "The return type of 'pauli_basis' will change from PauliTable to PauliList in a "
            "future release of Qiskit Terra.  Returning PauliTable is deprecated as of "
            "Qiskit Terra 0.19, and will be removed in a future release.  To immediately switch "
            "to the new behaviour, pass the keyword argument 'pauli_list=True'.",
            FutureWarning,
            stacklevel=2,
        )
        pauli_1q = PauliTable(
            np.array([[False, False], [True, False], [True, True], [False, True]], dtype=bool)
        )
    if num_qubits == 1:
        return pauli_1q
    pauli = pauli_1q
    for _ in range(num_qubits - 1):
        pauli = pauli_1q.tensor(pauli)
    if weight:
        return pauli.sort(weight=True)
    return pauli
