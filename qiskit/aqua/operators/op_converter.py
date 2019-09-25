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

""" op converter """

# pylint: disable=cyclic-import

import itertools
import logging
import sys

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.tools.parallel import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import AquaError, aqua_globals
from .weighted_pauli_operator import WeightedPauliOperator
from .matrix_operator import MatrixOperator
from .tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator

logger = logging.getLogger(__name__)


def _conversion(basis, matrix):
    pauli = Pauli.from_label(''.join(basis))
    trace_value = np.sum(matrix.dot(pauli.to_spmatrix()).diagonal())
    return trace_value, pauli


def to_weighted_pauli_operator(operator):
    """
    Converting a given operator to `WeightedPauliOperator`

    Args:
        operator (WeightedPauliOperator | TPBGroupedWeightedPauliOperator | MatrixOperator):
            one of supported operator type
    Returns:
        WeightedPauliOperator: the converted weighted pauli operator
    Raises:
        AquaError: Unsupported type to convert

    Warnings:
        Converting time from a MatrixOperator to a Pauli-type Operator grows exponentially.
        If you are converting a system with large number of qubits, it will take time.
        You can turn on DEBUG logging to check the progress.
    """
    if operator.__class__ == WeightedPauliOperator:
        return operator
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        # destroy the grouping but keep z2 symmetries info
        return WeightedPauliOperator(paulis=operator.paulis, z2_symmetries=operator.z2_symmetries,
                                     name=operator.name)
    elif operator.__class__ == MatrixOperator:
        if operator.is_empty():
            return WeightedPauliOperator(paulis=[])
        if operator.num_qubits > 10:
            logger.warning("Converting time from a MatrixOperator to a Pauli-type Operator grows "
                           "exponentially. If you are converting a system with large number of "
                           "qubits, it will take time. And now you are converting a %s-qubit "
                           "Hamiltonian. You can turn on DEBUG logging to check the progress."
                           "", operator.num_qubits)
        num_qubits = operator.num_qubits
        coeff = 2 ** (-num_qubits)

        paulis = []
        possible_basis = 'IXYZ'
        if operator.dia_matrix is not None:
            possible_basis = 'IZ'

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Converting a MatrixOperator to a Pauli-type Operator:")
            TextProgressBar(sys.stderr)
        results = parallel_map(_conversion,
                               list(itertools.product(possible_basis, repeat=num_qubits)),
                               task_kwargs={"matrix": operator._matrix},
                               num_processes=aqua_globals.num_processes)
        for trace_value, pauli in results:
            weight = trace_value * coeff
            if weight != 0.0 and np.abs(weight) > operator.atol:
                paulis.append([weight, pauli])

        return WeightedPauliOperator(paulis, z2_symmetries=operator.z2_symmetries,
                                     name=operator.name)
    else:
        raise AquaError("Unsupported type to convert to WeightedPauliOperator: "
                        "{}".format(operator.__class__))


def to_matrix_operator(operator):
    """
    Converting a given operator to `MatrixOperator`

    Args:
        operator (WeightedPauliOperator | TPBGroupedWeightedPauliOperator | MatrixOperator):
            one of supported operator type
    Returns:
        MatrixOperator: the converted matrix operator
    Raises:
        AquaError: Unsupported type to convert
    """
    if operator.__class__ == WeightedPauliOperator:
        if operator.is_empty():
            return MatrixOperator(None)
        hamiltonian = 0
        for weight, pauli in operator.paulis:
            hamiltonian += weight * pauli.to_spmatrix()
        return MatrixOperator(matrix=hamiltonian, z2_symmetries=operator.z2_symmetries,
                              name=operator.name)
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        op = WeightedPauliOperator(paulis=operator.paulis, z2_symmetries=operator.z2_symmetries,
                                   name=operator.name)
        return to_matrix_operator(op)
    elif operator.__class__ == MatrixOperator:
        return operator
    else:
        raise AquaError("Unsupported type to convert to MatrixOperator: "
                        "{}".format(operator.__class__))


# pylint: disable=invalid-name
def to_tpb_grouped_weighted_pauli_operator(operator, grouping_func, **kwargs):
    """

    Args:
        operator (Union(WeightedPauliOperator, TPBGroupedWeightedPauliOperator, MatrixOperator)):
            one of supported operator type
        grouping_func (Callable): a callable function that grouped the paulis in the operator.
        kwargs: other setting for `grouping_func` function

    Returns:
        TPBGroupedWeightedPauliOperator: the converted tensor-product-basis grouped weighted
                                         pauli operator
    Raises:
        AquaError: Unsupported type to convert
    """
    if operator.__class__ == WeightedPauliOperator:
        return grouping_func(operator, **kwargs)
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        # different tpb grouping approach is asked
        if grouping_func != operator.grouping_func and kwargs != operator.kwargs:
            return grouping_func(operator, **kwargs)
        else:
            return operator
    elif operator.__class__ == MatrixOperator:
        op = to_weighted_pauli_operator(operator)
        return grouping_func(op, **kwargs)
    else:
        raise AquaError("Unsupported type to convert to TPBGroupedWeightedPauliOperator: "
                        "{}".format(operator.__class__))
