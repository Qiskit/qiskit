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

"""Quadratic form."""

from typing import Union, Optional, List

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterExpression
from qiskit.circuit.library import QFT


class QuadraticForm(QuantumCircuit):
    """Converts an optimization problem (QUBO) to a negative value oracle.

    In addition, a state preparation operator is generated from the coefficients and constant of a
    QUBO, which can be used to encode the function into a quantum state. In conjunction, this oracle
    and operator can be used to flag the negative values of a QUBO encoded in a quantum state.

    The construction of the oracle is discussed in [1].

    References:
        [1]: Gilliam et al., Grover Adaptive Search for Constrained Polynomial Binary Optimization.
            arxiv:1912.04088.
    """

    def __init__(self,
                 num_value_qubits: int,
                 quadratic: Optional[Union[np.ndarray,
                                           List[List[Union[float, ParameterExpression]]]]] = None,
                 linear: Optional[Union[np.ndarray,
                                        List[Union[float, ParameterExpression]]]] = None,
                 offset: Optional[Union[float, ParameterExpression]] = None) -> None:
        """ """
        # check inputs match
        if quadratic is not None and linear is not None:
            if len(quadratic) != len(linear):
                raise ValueError('Mismatching sizes of quadratic and linear.')
            num_key = len(linear)
        elif quadratic is None and linear is not None:
            num_key = len(linear)
        elif quadratic is not None and linear is None:
            num_key = len(quadratic)
        else:  # both None
            num_key = 1

        qr_key = QuantumRegister(num_key)
        qr_value = QuantumRegister(num_value_qubits)
        super().__init__(qr_key, qr_value, name='q(x)')

        scaling = np.pi * 2 ** (1 - num_value_qubits)

        # constant coefficient
        if offset is not None:
            for i in range(num_value_qubits):
                self.u1(scaling * 2 ** i * offset, qr_value[i])

        if linear is not None:
            for j in range(num_key):
                value = linear[j]
                if value != 0:
                    for i in range(num_value_qubits):
                        self.cu1(scaling * 2 ** i * value, qr_key[j], qr_value[i])

        if quadratic is not None:
            for j in range(num_key):
                for k in range(j + 1, num_key):
                    value = quadratic[j][k]
                    if value != 0:
                        for i in range(num_value_qubits):
                            self.mcu1(scaling * 2 ** i * value, [qr_key[j], qr_key[k]], qr_value[i])

        # Add IQFT. Adding swaps at the end of the IQFT, not the beginning.
        iqft = QFT(num_value_qubits, do_swaps=False).inverse()
        self.append(iqft, qr_value)

        for i in range(num_value_qubits // 2):
            self.swap(qr_value[i], qr_value[-(i + 1)])
