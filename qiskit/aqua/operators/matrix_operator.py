# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from copy import deepcopy
from functools import reduce
import logging
import warnings

import numpy as np
from scipy import sparse as scisparse
from scipy import linalg as scila
from qiskit import QuantumCircuit

from qiskit.aqua.operators.base_operator import BaseOperator
from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


class MatrixOperator(BaseOperator):

    """
    Operators relevant for quantum applications

    Note:
        For grouped paulis representation, all operations will always convert it to paulis and then convert it back.
        (It might be a performance issue.)
    """

    def __init__(self, matrix, basis=None, z2_symmetries=None, atol=1e-12, name=None):
        """
        Args:
            matrix (numpy.ndarray or scipy.sparse.csr_matrix): a 2-D sparse matrix represents operator
                                                               (using CSR format internally)
        """
        super().__init__(basis, z2_symmetries, name)
        if matrix is not None:
            matrix = matrix if scisparse.issparse(matrix) else scisparse.csr_matrix(matrix)
            matrix = matrix if scisparse.isspmatrix_csr(matrix) else matrix.to_csr(copy=True)
        self._matrix = matrix
        self._atol = atol

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, new_value):
        self._atol = new_value

    def add(self, other, copy=False):
        out = self.copy() if copy else self
        out._matrix += other._matrix
        return out

    def sub(self, other, copy=False):
        out = self.copy() if copy else self
        out._matrix -= other._matrix
        return out

    def __add__(self, other):
        """Overload + operation"""
        return self.add(other, copy=True)

    def __iadd__(self, other):
        """Overload += operation"""
        return self.add(other, copy=False)

    def __sub__(self, other):
        """Overload - operation"""
        return self.sub(other, copy=True)

    def __isub__(self, other):
        """Overload -= operation"""
        return self.sub(other, copy=False)

    def __neg__(self):
        """Overload unary - """
        out = self.copy()
        out._matrix *= -1.0
        return out

    def __eq__(self, other):
        """Overload == operation"""
        return np.all(self._matrix == other.matrix)

    def __str__(self):
        """Overload str()"""
        curr_repr = 'matrix'
        length = "{}x{}".format(2 ** self.num_qubits, 2 ** self.num_qubits)
        ret = "Representation: {}, qubits: {}, size: {}".format(curr_repr, self.num_qubits, length)
        return ret

    def copy(self):
        """Get a copy of self."""
        return deepcopy(self)

    def chop(self, threshold=None, copy=False):
        """
        Eliminate the real and imagine part of coeff in each pauli by `threshold`.
        If pauli's coeff is less then `threshold` in both real and imagine parts, the pauli is removed.
        To align the internal representations, all available representations are chopped.
        The chopped result is stored back to original property.
        Note: if coeff is real-only, the imag part is skipped.

        Args:
            threshold (float): threshold chops the paulis
        """
        threshold = self._atol if threshold is None else threshold

        def chop_real_imag(coeff):
            temp_real = coeff.real if np.absolute(coeff.real) >= threshold else 0.0
            temp_imag = coeff.imag if np.absolute(coeff.imag) >= threshold else 0.0
            if temp_real == 0.0 and temp_imag == 0.0:
                return 0.0
            else:
                new_coeff = temp_real + 1j * temp_imag
                return new_coeff

        op = self.copy() if copy else self

        rows, cols = op._matrix.nonzero()
        for row, col in zip(rows, cols):
            op._matrix[row, col] = chop_real_imag(op._matrix[row, col])
        op._matrix.eliminate_zeros()
        return op

    def _scaling_weight(self, scaling_factor):
        # TODO: existed for supporting the deprecated method, will remove it.
        self._matrix = scaling_factor * self._matrix

    def __mul__(self, other):
        """
        Overload * operation. Only support two Operators have the same representation mode.

        Returns:
            MatrixOperator: the multipled Operator.

        Raises:
            TypeError, if two Operators do not have the same representations.
        """
        ret_matrix = self._matrix.dot(other.matrix)
        return MatrixOperator(matrix=ret_matrix)

    @property
    def dia_matrix(self):
        dia_matrix = self._matrix.diagonal()
        if not scisparse.csr_matrix(dia_matrix).nnz == self._matrix.nnz:
            dia_matrix = None
        return dia_matrix

    @property
    def matrix(self):
        """Getter of matrix."""
        return self._matrix if self.dia_matrix is None else self.dia_matrix

    @property
    def dense_matrix(self):
        """Getter of matrix in dense matrix form."""
        return self._matrix.toarray()

    @property
    def num_qubits(self):
        """
        number of qubits required for the operator.

        Returns:
            int: number of qubits
        """
        if self.is_empty():
            logger.warning("Operator is empty, Return 0.")
            return 0
        return int(np.log2(self._matrix.shape[0]))

    def print_details(self):
        """
        Returns:
            str: a formated operator.
        """
        ret = str(self._matrix)
        return ret

    def construct_evaluation_circuit(self, operator_mode=None, input_circuit=None, backend=None, qr=None, cr=None,
                                     use_simulator_operator_mode=False, wave_function=None, statevector_mode=None,
                                     circuit_name_prefix=''):
        """
        Construct the circuits for evaluation.

        Args:
            wave_function (QuantumCircuit): the quantum circuit.
            circuit_name_prefix (str, optional): a prefix of circuit name

        Returns:
            [QuantumCircuit]: the circuits for computing the expectation of the operator over
                              the wavefunction evaluation.
        """
        if operator_mode is not None:
            warnings.warn("operator_mode option is deprecated and it will be removed after 0.6, "
                          "Every operator knows which mode is using, not need to indicate the mode.",
                          DeprecationWarning)

        if input_circuit is not None:
            warnings.warn("input_circuit option is deprecated and it will be removed after 0.6, "
                          "Use `wave_function` instead.",
                          DeprecationWarning)
            wave_function = input_circuit
        else:
            if wave_function is None:
                raise AquaError("wave_function must not be None.")

        if backend is not None:
            warnings.warn("backend option is deprecated and it will be removed after 0.6, "
                          "No need for backend when using matrix operator",
                          DeprecationWarning)

        return [wave_function.copy(name=circuit_name_prefix + 'psi')]

    def evaluate_with_result(self, operator_mode=None, circuits=None, backend=None, result=None,
                             use_simulator_operator_mode=False, statevector_mode=None,
                             circuit_name_prefix=''):
        """
        Use the executed result with operator to get the evaluated value.

        Args:
            result (qiskit.Result): the result from the backend.
            circuit_name_prefix (str, optional): a prefix of circuit name

        Returns:
            float: the mean value
            float: the standard deviation
        """
        if operator_mode is not None:
            warnings.warn("operator_mode option is deprecated and it will be removed after 0.6, "
                          "Every operator knows which mode is using, not need to indicate the mode.",
                          DeprecationWarning)
        if circuits is not None:
            warnings.warn("circuits option is deprecated and it will be removed after 0.6, "
                          "we will retrieve the circuit via its unique name directly.",
                          DeprecationWarning)
        if backend is not None:
            warnings.warn("backend option is deprecated and it will be removed after 0.6, "
                          "No need for backend when using matrix operator",
                          DeprecationWarning)

        avg, std_dev = 0.0, 0.0
        quantum_state = np.asarray(result.get_statevector(circuit_name_prefix + 'psi'))
        avg = np.vdot(quantum_state, self._matrix.dot(quantum_state))

        return avg, std_dev

    def evaluate_with_statevector(self, quantum_state):
        """

        Args:
            quantum_state (numpy.ndarray):

        Returns:
            float: the mean value
            float: the standard deviation
        """
        avg = np.vdot(quantum_state, self._matrix.dot(quantum_state))
        return avg, 0.0

    @staticmethod
    def _suzuki_expansion_slice_matrix(pauli_list, lam, expansion_order):
        """
        Compute the matrix for a single slice of the suzuki expansion following the paper
        https://arxiv.org/pdf/quant-ph/0508139.pdf

        Args:
            pauli_list (list): The operator's complete list of pauli terms for the suzuki expansion
            lam (complex): The parameter lambda as defined in said paper
            expansion_order (int): The order for the suzuki expansion

        Returns:
            numpy array: The matrix representation corresponding to the specified suzuki expansion
        """
        # pylint: disable=no-member
        if expansion_order == 1:
            left = reduce(
                lambda x, y: x @ y,
                [scila.expm(lam / 2 * c * p.to_spmatrix().tocsc()) for c, p in pauli_list]
            )
            right = reduce(
                lambda x, y: x @ y,
                [scila.expm(lam / 2 * c * p.to_spmatrix().tocsc()) for c, p in reversed(pauli_list)]
            )
            return left @ right
        else:
            pk = (4 - 4 ** (1 / (2 * expansion_order - 1))) ** -1
            side_base = MatrixOperator._suzuki_expansion_slice_matrix(
                pauli_list,
                lam * pk,
                expansion_order - 1
            )
            side = side_base @ side_base
            middle = MatrixOperator._suzuki_expansion_slice_matrix(
                pauli_list,
                lam * (1 - 4 * pk),
                expansion_order - 1
            )
            return side @ middle @ side

    def evolve(self, state_in, evo_time=0, evo_mode=None, num_time_slices=0, quantum_registers=None,
               expansion_mode='trotter', expansion_order=1):
        """
        Carry out the eoh evolution for the operator under supplied specifications.

        Args:
            state_in: The initial state for the evolution
            evo_time (int): The evolution time
            num_time_slices (int): The number of time slices for the expansion
            expansion_mode (str): The mode under which the expansion is to be done.
                Currently support 'trotter', which follows the expansion as discussed in
                http://science.sciencemag.org/content/273/5278/1073,
                and 'suzuki', which corresponds to the discussion in
                https://arxiv.org/pdf/quant-ph/0508139.pdf
            expansion_order (int): The order for suzuki expansion

        Returns:
            Return the matrix vector multiplication result.
        """
        if evo_mode is not None:
            warnings.warn("evo_mode option is deprecated and it will be removed after 0.6, "
                          "Every operator knows which mode is using, not need to indicate the mode.",
                          DeprecationWarning)

        if quantum_registers is not None:
            warnings.warn("quantum_registers option is not required by `MatrixOperator` and it will be removed "
                          "after 0.6.",
                          DeprecationWarning)

        from .op_converter import to_weighted_pauli_operator
        # pylint: disable=no-member
        if num_time_slices < 0 or not isinstance(num_time_slices, int):
            raise ValueError('Number of time slices should be a non-negative integer.')
        if expansion_mode not in ['trotter', 'suzuki']:
            raise ValueError('Expansion mode {} not supported.'.format(expansion_mode))

        if num_time_slices == 0:
            return scila.expm(-1.j * evo_time * self._matrix.tocsc()) @ state_in
        else:

            pauli_op = to_weighted_pauli_operator(self)
            pauli_list = pauli_op.reorder_paulis()

            if len(pauli_list) == 1:
                approx_matrix_slice = scila.expm(
                    -1.j * evo_time / num_time_slices * pauli_list[0][0] * pauli_list[0][1].to_spmatrix().tocsc()
                )
            else:
                if expansion_mode == 'trotter':
                    approx_matrix_slice = reduce(
                        lambda x, y: x @ y,
                        [
                            scila.expm(-1.j * evo_time / num_time_slices * c * p.to_spmatrix().tocsc())
                            for c, p in pauli_list
                        ]
                    )
                # suzuki expansion
                elif expansion_mode == 'suzuki':
                    approx_matrix_slice = MatrixOperator._suzuki_expansion_slice_matrix(
                        pauli_list,
                        -1.j * evo_time / num_time_slices,
                        expansion_order
                    )
                else:
                    raise ValueError('Unrecognized expansion mode {}.'.format(expansion_mode))
            return reduce(lambda x, y: x @ y, [approx_matrix_slice] * num_time_slices) @ state_in

    def is_empty(self):
        """
        Check Operator is empty or not.

        Returns:
            bool: is empty?
        """
        if self._matrix is None or self._matrix.nnz == 0:
            return True
        else:
            return False
