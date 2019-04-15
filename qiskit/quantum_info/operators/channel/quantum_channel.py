# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint:disable=abstract-method
"""
Abstract base class for Quantum Channels.
"""

from abc import ABC

import numpy as np

from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _to_operator


class QuantumChannel(BaseOperator, ABC):
    """Quantum channel representation base class."""

    def is_cptp(self):
        """Return True if completely-positive trace-preserving."""
        choi = _to_choi(self.rep, self._data, *self.dim)
        return self._is_cp(choi) and self._is_tp(choi)

    def is_unitary(self):
        """Return True if QuantumChannel is a unitary channel."""
        try:
            op = self.to_operator()
            return op.is_unitary()
        except QiskitError:
            return False

    def to_operator(self):
        """Try to convert channel to a unitary representation Operator."""
        mat = _to_operator(self.rep, self._data, *self.dim)
        return Operator(mat, self.input_dims(), self.output_dims())

    def _is_cp(self, choi=None):
        """Test if a channel is completely-positive (CP)"""
        if choi is None:
            choi = _to_choi(self.rep, self._data, *self.dim)
        return is_positive_semidefinite_matrix(
            choi, rtol=self._rtol, atol=self._atol)

    def _is_tp(self, choi=None):
        """Test if Choi-matrix is trace-preserving (TP)"""
        if choi is None:
            choi = _to_choi(self.rep, self._data, *self.dim)
        # Check if the partial trace is the identity matrix
        d_in, d_out = self.dim
        mat = np.trace(
            np.reshape(choi, (d_in, d_out, d_in, d_out)), axis1=1, axis2=3)
        return is_identity_matrix(mat, rtol=self._rtol, atol=self._atol)

    def _format_state(self, state, density_matrix=False):
        """Format input state so it is statevector or density matrix"""
        state = np.array(state)
        shape = state.shape
        ndim = state.ndim
        if ndim > 2:
            raise QiskitError('Input state is not a vector or matrix.')
        # Flatten column-vector to vector
        if ndim == 2:
            if shape[1] != 1 and shape[1] != shape[0]:
                raise QiskitError('Input state is not a vector or matrix.')
            if shape[1] == 1:
                # flatten colum-vector to vector
                state = np.reshape(state, shape[0])
        # Convert statevector to density matrix if required
        if density_matrix and ndim == 1:
            state = np.outer(state, np.transpose(np.conj(state)))
        return state
