# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
DEPRECATED Tools for working with Pauli Operators.
"""

from warnings import warn
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.pauli import Pauli


def pauli_group(number_of_qubits, case="weight"):
    """DEPRECATED: Return the Pauli group with 4^n elements.

    This function is deprecated. Use :func:`~qiskit.quantum_info.pauli_basis`
    for equivalent functionality.

    The phases have been removed.
    case 'weight' is ordered by Pauli weights and
    case 'tensor' is ordered by I,X,Y,Z counting lowest qubit fastest.

    Args:
        number_of_qubits (int): number of qubits
        case (str): determines ordering of group elements ('weight' or 'tensor')

    Returns:
        list: list of Pauli objects

    Raises:
        QiskitError: case is not 'weight' or 'tensor'
        QiskitError: number_of_qubits is larger than 4
    """
    warn(
        "`insert_paulis` is deprecated and will be removed no earlier than "
        "3 months after the release date. For equivalent functionality to"
        "`qiskit.quantum_info.pauli_group` instead. "
        "`pauli_group(n)` is equivalent to `pauli_basis(n, weight=True)`, "
        '`pauli_group(n, case="tensor") is equivalent to `pauli_basis(n)`',
        DeprecationWarning,
        stacklevel=2,
    )
    if number_of_qubits < 5:
        temp_set = []

        if case == "weight":
            tmp = pauli_group(number_of_qubits, case="tensor")
            # sort on the weight of the Pauli operator
            return sorted(tmp, key=lambda x: -np.count_nonzero(np.array(x.to_label(), "c") == b"I"))
        elif case == "tensor":
            # the Pauli set is in tensor order II IX IY IZ XI ...
            for k in range(4 ** number_of_qubits):
                z = np.zeros(number_of_qubits, dtype=bool)
                x = np.zeros(number_of_qubits, dtype=bool)
                # looping over all the qubits
                for j in range(number_of_qubits):
                    # making the Pauli for each j fill it in from the
                    # end first
                    element = (k // (4 ** j)) % 4
                    if element == 1:
                        x[j] = True
                    elif element == 2:
                        z[j] = True
                        x[j] = True
                    elif element == 3:
                        z[j] = True
                temp_set.append(Pauli(z, x))
            return temp_set
        else:
            raise QiskitError(
                "Only support 'weight' or 'tensor' cases " "but you have {}.".format(case)
            )

    raise QiskitError("Only support number of qubits is less than 5")
