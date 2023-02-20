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
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList


def pauli_basis(num_qubits, weight=False, pauli_list=None):
    """Return the ordered PauliTable or PauliList for the n-qubit Pauli basis.

    Args:
        num_qubits (int): number of qubits
        weight (bool): if True optionally return the basis sorted by Pauli weight
                       rather than lexicographic order (Default: False)
        pauli_list (bool): [Deprecated] This argument is deprecated and remains
                           for backwards compatability. It has no effect.

    Returns:
        PauliList: the Paulis for the basis
    """
    if pauli_list is not None:
        warnings.warn(
            "The `pauli_list` kwarg is deprecated as of Qiskit Terra 0.22 and "
            "no longer has an effect as `pauli_basis` always returns a PauliList.",
            DeprecationWarning,
            stacklevel=2,
        )
    pauli_1q = PauliList(["I", "X", "Y", "Z"])
    if num_qubits == 1:
        return pauli_1q
    pauli = pauli_1q
    for _ in range(num_qubits - 1):
        pauli = pauli_1q.tensor(pauli)
    if weight:
        return pauli.sort(weight=True)
    return pauli
