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

from __future__ import annotations
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit.utils.deprecation import deprecate_arg


@deprecate_arg(
    "pauli_basis",
    since="0.22",
    additional_msg=(
        "The argument ``pauli_list`` has no effect as the function always returns a PauliList."
    ),
)
def pauli_basis(num_qubits: int, weight: bool = False, pauli_list=None) -> PauliList:
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
    del pauli_list
    pauli_1q = PauliList(["I", "X", "Y", "Z"])
    if num_qubits == 1:
        return pauli_1q
    pauli = pauli_1q
    for _ in range(num_qubits - 1):
        pauli = pauli_1q.tensor(pauli)
    if weight:
        return pauli.sort(weight=True)
    return pauli
