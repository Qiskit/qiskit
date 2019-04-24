# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint:disable=abstract-method
"""
Abstract base class for Quantum Channels.
"""

import numpy as np

from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _to_kraus
from qiskit.quantum_info.operators.channel.transformations import _to_operator


class QuantumChannel(BaseOperator):
    """Quantum channel representation base class."""

    def is_cptp(self, atol=None, rtol=None):
        """Return True if completely-positive trace-preserving (CPTP)."""
        choi = _to_choi(self.rep, self._data, *self.dim)
        return self._is_cp_helper(choi, atol, rtol) and self._is_tp_helper(
            choi, atol, rtol)

    def is_tp(self, atol=None, rtol=None):
        """Test if a channel is completely-positive (CP)"""
        choi = _to_choi(self.rep, self._data, *self.dim)
        return self._is_tp_helper(choi, atol, rtol)

    def is_cp(self, atol=None, rtol=None):
        """Test if Choi-matrix is completely-positive (CP)"""
        choi = _to_choi(self.rep, self._data, *self.dim)
        return self._is_cp_helper(choi, atol, rtol)

    def is_unitary(self, atol=None, rtol=None):
        """Return True if QuantumChannel is a unitary channel."""
        try:
            op = self.to_operator()
            return op.is_unitary(atol=atol, rtol=rtol)
        except QiskitError:
            return False

    def to_operator(self):
        """Try to convert channel to a unitary representation Operator."""
        mat = _to_operator(self.rep, self._data, *self.dim)
        return Operator(mat, self.input_dims(), self.output_dims())

    def _is_cp_helper(self, choi, atol, rtol):
        """Test if a channel is completely-positive (CP)"""
        if atol is None:
            atol = self._atol
        if rtol is None:
            rtol = self._rtol
        return is_positive_semidefinite_matrix(choi, rtol=rtol, atol=atol)

    def _is_tp_helper(self, choi, atol, rtol):
        """Test if Choi-matrix is trace-preserving (TP)"""
        if atol is None:
            atol = self._atol
        if rtol is None:
            rtol = self._rtol
        # Check if the partial trace is the identity matrix
        d_in, d_out = self.dim
        mat = np.trace(
            np.reshape(choi, (d_in, d_out, d_in, d_out)), axis1=1, axis2=3)
        return is_identity_matrix(mat, rtol=rtol, atol=atol)

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

    @classmethod
    def _init_transformer(cls, data):
        """Convert input into a QuantumChannel subclass object or Operator object"""
        if issubclass(data.__class__, QuantumChannel):
            return data
        # Use to_channel method to convert to channel
        if hasattr(data, 'to_channel'):
            # Use to_channel method to convert to channel
            return data.to_channel()
        # If no to_channel method try converting to a matrix Operator
        # which can be transformed into a channel
        return Operator(data)
