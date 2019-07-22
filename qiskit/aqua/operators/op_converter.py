# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import itertools
import warnings

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua import AquaError
from qiskit.aqua import Operator
from .weighted_pauli_operator import WeightedPauliOperator
from .matrix_operator import MatrixOperator
from .tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator


def to_weighted_pauli_operator(operator):
    """
    Converting a given operator to `WeightedPauliOperator`

    Args:
        operator (WeightedPauliOperator | TPBGroupedWeightedPauliOperator | MatrixOperator | Operator):
            one of supported operator type
    Returns:
        WeightedPauliOperator: the converted weighted pauli operator
    """
    if operator.__class__ == WeightedPauliOperator:
        return operator
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        # destroy the grouping but keep z2 symmetries info
        return WeightedPauliOperator(paulis=operator.paulis, z2_symmetries=operator.z2_symmetries, name=operator.name)
    elif operator.__class__ == MatrixOperator:
        if operator.is_empty():
            return WeightedPauliOperator(paulis=[])

        num_qubits = operator.num_qubits
        coeff = 2 ** (-num_qubits)

        paulis = []
        possible_basis = 'IXYZ'
        if operator.dia_matrix is not None:
            possible_basis = 'IZ'
        # generate all possible paulis basis
        for basis in itertools.product(possible_basis, repeat=num_qubits):
            pauli = Pauli.from_label(''.join(basis))
            trace_value = np.sum(operator._matrix.dot(pauli.to_spmatrix()).diagonal())
            weight = trace_value * coeff
            if weight != 0.0:
                paulis.append([weight, pauli])

        return WeightedPauliOperator(paulis, z2_symmetries=operator.z2_symmetries, name=operator.name)
    elif operator.__class__ == Operator:
        warnings.warn("The `Operator` class is deprecated. Please use `WeightedPauliOperator` or "
                      "`TPBGroupedWeightedPauliOperator` or `MatrixOperator` instead", DeprecationWarning)
        return operator.to_weighted_pauli_operator()
    else:
        raise AquaError("Unsupported type to convert to WeightedPauliOperator: {}".format(operator.__class__))


def to_matrix_operator(operator):
    """
    Converting a given operator to `WeightedPauliOperator`

    Args:
        operator (WeightedPauliOperator | TPBGroupedWeightedPauliOperator | MatrixOperator | Operator):
            one of supported operator type
    Returns:
        MatrixOperator: the converted matrix operator
    """
    if operator.__class__ == WeightedPauliOperator:
        if operator.is_empty():
            return MatrixOperator(None)
        hamiltonian = 0
        for weight, pauli in operator.paulis:
            hamiltonian += weight * pauli.to_spmatrix()
        return MatrixOperator(matrix=hamiltonian, z2_symmetries=operator.z2_symmetries, name=operator.name)
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        # destroy the grouping but keep z2 symmetries info
        return WeightedPauliOperator(paulis=operator.paulis, z2_symmetries=operator.z2_symmetries, name=operator.name)
    elif operator.__class__ == MatrixOperator:
        return operator
    elif operator.__class__ == Operator:
        warnings.warn("The `Operator` class is deprecated. Please use `WeightedPauliOperator` or "
                      "`TPBGroupedWeightedPauliOperator` or `MatrixOperator` instead", DeprecationWarning)
        return operator.to_matrix_operator()
    else:
        raise AquaError("Unsupported type to convert to WeightedPauliOperator: {}".format(operator.__class__))


def to_tpb_grouped_weighted_pauli_operator(operator, grouping_func, **kwargs):
    """

    Args:
        operator (WeightedPauliOperator | TPBGroupedWeightedPauliOperator | MatrixOperator | Operator):
            one of supported operator type
        grouping_func (Callable): a callable function that grouped the paulis in the operator.
        kwargs: other setting for `grouping_func` function

    Returns:
        TPBGroupedWeightedPauliOperator: the converted tesnor-product-basis grouped weighted pauli operator
    """
    if operator.__class__ == WeightedPauliOperator:
        return grouping_func(operator, **kwargs)
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        # different tpb grouning approach is asked
        if grouping_func != operator.grouping_func and kwargs != operator.kwargs:
            return grouping_func(operator, **kwargs)
        else:
            return operator
    elif operator.__class__ == MatrixOperator:
        op = to_weighted_pauli_operator(operator)
        return grouping_func(op, **kwargs)
    elif operator.__class__ == Operator:
        warnings.warn("The `Operator` class is deprecated. Please use `WeightedPauliOperator` or "
                      "`TPBGroupedWeightedPauliOperator` or `MatrixOperator` instead", DeprecationWarning)
        return operator.to_tpb_grouped_weighted_pauli_operator()
    else:
        raise AquaError("Unsupported type to convert to WeightedPauliOperator: {}".format(operator.__class__))
