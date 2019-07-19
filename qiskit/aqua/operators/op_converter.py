
import itertools

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua import AquaError
from .weighted_pauli_operator import WeightedPauliOperator
from .matrix_operator import MatrixOperator
from .tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator


def to_weighted_pauli_operator(operator, name=None):
    """
    Converting a given operator to `WeightedPauliOperator`

    Args:
        operator (WeightedPauliOperator | TPBGroupedWeightedPauliOperator | MatrixOperator): one of supported operator
                                                                                             type
    Returns:
        WeightedPauliOperator
    """
    if operator.__class__ == WeightedPauliOperator:
        return operator
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        # destory the grouping but keep z2 symmetris info
        return WeightedPauliOperator(paulis=operator.paulis, z2_symmetries=operator.z2_symmetries)
    elif operator.__class__ == MatrixOperator:
        return _from_matrix_to_weighted_pauli(operator)
    else:
        raise AquaError("Unsupported type to convert to WeightedPauliOperator: {}".format(operator.__class__))


def to_matrix_operator(operator, name=None):
    if operator.__class__ == WeightedPauliOperator:
        return _from_weighted_pauli_to_matrix(operator)
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        # destory the grouping but keep z2 symmetris info
        return WeightedPauliOperator(paulis=operator.paulis, z2_symmetries=operator.z2_symmetries)
    elif operator.__class__ == MatrixOperator:
        return operator
    else:
        raise AquaError("Unsupported type to convert to WeightedPauliOperator: {}".format(operator.__class__))


def to_tpb_grouped_weighted_pauli_operator(operator, grouping_func, name=None, **kwargs):
    """

    Args:
        operator:
        grouping_func:
        kwargs:

    Returns:

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
        op = _from_matrix_to_weighted_pauli(operator)
        return grouping_func(op, **kwargs)
    else:
        raise AquaError("Unsupported type to convert to WeightedPauliOperator: {}".format(operator.__class__))


def _from_weighted_pauli_to_matrix(operator):
    """

    Args:
        operator (WeightedPauliOperator):

    Returns:

    """
    if operator.is_empty():
        return MatrixOperator(None)

    hamiltonian = 0
    for weight, pauli in operator.paulis:
        hamiltonian += weight * pauli.to_spmatrix()
    return MatrixOperator(matrix=hamiltonian, z2_symmetries=operator.z2_symmetries, name=operator.name)


def _from_matrix_to_weighted_pauli(operator):
    """
    Convert matrix to paulis

    Note:
        Conversion from Paulis to matrix: H = sum_i alpha_i * Pauli_i
        Conversion from matrix to Paulis: alpha_i = coeff * Trace(H.Pauli_i) (dot product of trace)
            where coeff = 2^(- # of qubits), # of qubit = log2(dim of matrix)

    Args:
        operator (MatrixOperator)

    Returns:
        WeightedPauliOperator:
    """
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
