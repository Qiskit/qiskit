# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A circuit implementing a quadratic form on binary variables."""

from typing import Union, Optional, List

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterExpression
from ..basis_change import QFT


class QuadraticForm(QuantumCircuit):
    r"""Implements a quadratic form on binary variables encoded in qubit registers.

    A quadratic form on binary variables is a quadratic function :math:`Q` acting on a binary
    variable of :math:`n` bits, :math:`x = x_0 ... x_{n-1}`. For a matrix :math:`A`, a vector
    :math:`b` and a scalar :math:`c` the function can be written as

    .. math::

        Q(x) = x^T A x + x^T b + c

    Provided with :math:`m` qubits to encode the value, this circuit computes :math:`Q(x) \mod 2^m`.

    .. math::

        |x\rangle_n |0\rangle_m \mapsto |x\rangle_n |Q(x) \mod 2\rangle_m

    The implementation of this circuit is discussed in [1], Fig. 6.

    References:
        [1]: Gilliam et al., Grover Adaptive Search for Constrained Polynomial Binary Optimization.
             `arXiv:1912.04088 <https://arxiv.org/pdf/1912.04088.pdf>`_

    """

    def __init__(self,
                 num_result_qubits: int,
                 quadratic: Optional[Union[np.ndarray,
                                           List[List[Union[float, ParameterExpression]]]]] = None,
                 linear: Optional[Union[np.ndarray,
                                        List[Union[float, ParameterExpression]]]] = None,
                 offset: Optional[Union[float, ParameterExpression]] = None) -> None:
        r"""
        Args:
            num_result_qubits: The number of qubits to encode the result. Called :math:`m` in
                the class documentation.
            quadratic: A matrix containing the quadratic coefficients, :math:`A`.
            linear: An array containing the linear coefficients, :math:`b`.
            offset: A constant offset, :math:`c`.

        Raises:
            ValueError: If ``linear`` and ``quadratic`` have mismatching sizes.
        """
        # check inputs match
        if quadratic is not None and linear is not None:
            if len(quadratic) != len(linear):
                raise ValueError('Mismatching sizes of quadratic and linear.')
            num_input_qubits = len(linear)
        elif quadratic is None and linear is not None:
            num_input_qubits = len(linear)
        elif quadratic is not None and linear is None:
            num_input_qubits = len(quadratic)
        else:  # both None
            num_input_qubits = 1

        qr_input = QuantumRegister(num_input_qubits)
        qr_result = QuantumRegister(num_result_qubits)
        super().__init__(qr_input, qr_result, name='Q(x)')

        scaling = np.pi * 2 ** (1 - num_result_qubits)

        # constant coefficient
        if offset is not None and offset != 0:
            for i in range(num_result_qubits):
                self.u1(scaling * 2 ** i * offset, qr_result[i])

        # the linear part consists of the vector and the diagonal of the
        # matrix, since x_i * x_i = x_i, as x_i is a binary variable
        for j in range(num_input_qubits):
            value = linear[j] if linear is not None else 0
            value += quadratic[j][j] if quadratic is not None else 0
            if value != 0:
                for i in range(num_result_qubits):
                    self.cu1(scaling * 2 ** i * value, qr_input[j], qr_result[i])

        # the quadratic part adds A_ij and A_ji as x_i x_j == x_j x_i
        if quadratic is not None:
            for j in range(num_input_qubits):
                for k in range(j + 1, num_input_qubits):
                    value = quadratic[j][k] + quadratic[k][j]
                    if value != 0:
                        for i in range(num_result_qubits):
                            self.mcu1(scaling * 2 ** i * value, [qr_input[j], qr_input[k]],
                                      qr_result[i])

        # add the inverse QFT, swaps are added at the end, not the beginning here
        iqft = QFT(num_result_qubits, do_swaps=False).inverse()
        self.append(iqft, qr_result)

        for i in range(num_result_qubits // 2):
            self.swap(qr_result[i], qr_result[-(i + 1)])
