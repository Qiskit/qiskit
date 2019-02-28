# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from numbers import Number

import numpy as np

from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
from .basechannel import QuantumChannel
from .choi import Choi
from .transformations import _to_kraus


class Kraus(QuantumChannel):
    """Kraus representation of a quantum channel."""

    def __init__(self, data, input_dim=None, output_dim=None):
        # Check if input is a quantum channel object
        # If so we disregard the dimension kwargs
        if issubclass(data.__class__, QuantumChannel):
            input_dim, output_dim = data.dims
            kraus = _to_kraus(data.rep, data._data, input_dim, output_dim)

        else:
            # Check if it is a single unitary matrix A for channel:
            # E(rho) = A * rho * A^\dagger
            if isinstance(data, np.ndarray) or np.array(data).ndim == 2:
                # Convert single Kraus op to general Kraus pair
                kraus = ([np.array(data, dtype=complex)], None)
                shape = kraus[0][0].shape

            # Check if single Kraus set [A_i] for channel:
            # E(rho) = sum_i A_i * rho * A_i^dagger
            elif isinstance(data, list) and len(data) > 0:
                # Get dimensions from first Kraus op
                kraus = [np.array(data[0], dtype=complex)]
                shape = kraus[0].shape
                # Iterate over remaining ops and check they are same shape
                for a in data[1:]:
                    op = np.array(a, dtype=complex)
                    if op.shape != shape:
                        raise QiskitError("Kraus operators are different dimensions.")
                    kraus.append(op)
                # Convert single Kraus set to general Kraus pair
                kraus = (kraus, None)

            # Check if generalized Kraus set ([A_i], [B_i]) for channel:
            # E(rho) = sum_i A_i * rho * B_i^dagger
            elif isinstance(data, tuple) and len(data) == 2 and len(data[0]) > 0:
                kraus_left = [np.array(data[0][0], dtype=complex)]
                shape = kraus_left[0].shape
                for a in data[0][1:]:
                    op = np.array(a, dtype=complex)
                    if op.shape != shape:
                        raise QiskitError("Kraus operators are different dimensions.")
                    kraus_left.append(op)
                if data[1] is None:
                    kraus = (kraus_left, None)
                else:
                    kraus_right = []
                    for b in data[1]:
                        op = np.array(b, dtype=complex)
                        if op.shape != shape:
                            raise QiskitError("Kraus operators are different dimensions.")
                        kraus_right.append(op)
                    kraus = (kraus_left, kraus_right)
            else:
                raise QiskitError("Invalid input for Kraus channel.")
        dout, din = kraus[0][0].shape
        if (input_dim and input_dim != din) or (output_dim and output_dim != dout):
            raise QiskitError("Invalid dimensions for Kraus input.")

        if kraus[1] is None or np.allclose(kraus[0], kraus[1]):
            # Standard Kraus map
            super().__init__('Kraus', (kraus[0], None), input_dim=din, output_dim=dout)
        else:
            # General (non-CPTP) Kraus map
            super().__init__('Kraus', kraus, input_dim=din, output_dim=dout)

    @property
    def data(self):
        """Return list of Kraus matrices for channel."""
        if self._data[1] is None:
            # If only a single Kraus set, don't return the tuple
            # Just the fist set
            return self._data[0]
        else:
            # Otherwise return the tuple of both kraus sets
            return self._data

    def evolve(self, state):
        """Apply the channel to a quantum state.

        Args:
            state (quantum_state like): A statevector or density matrix.

        Returns:
            A density matrix or statevector.
        """
        state = self._check_state(state)
        if state.ndim == 1 and self._data[1] is None and len(self._data[0]) == 1:
            # If we only have a single Kraus operator we can implement unitary-type
            # evolution of a state vector psi -> K[0].psi
            return np.dot(self._data[0][0], state)
        # Otherwise we always return a density matrix
        state = self._format_density_matrix(state)
        kraus_l, kraus_r = self._data
        if kraus_r is None:
            kraus_r = kraus_l
        return np.einsum('AiB,BC,AjC->ij', kraus_l, state, np.conjugate(kraus_r))

    def is_cptp(self, atol=ATOL_DEFAULT):
        """Test if the Kraus channel is a CPTP map."""
        if self._data[1] is not None:
            return False
        accum = 0j
        for op in self._data[0]:
            accum += np.dot(np.transpose(np.conj(op)), op)
        return is_identity_matrix(accum, atol=atol)

    def canonical(self, inplace=False):
        """Convert to canonical Kraus representation"""
        tmp = Kraus(Choi(self))
        if inplace:
            self._data = tmp._data
            return self
        return tmp

    def conjugate(self, inplace=False):
        """Return the conjugate channel"""
        kraus_l, kraus_r = self._data
        kraus_l = [k.conj() for k in kraus_l]
        if kraus_r is not None:
            kraus_r = [k.conj() for k in kraus_r]
        if inplace:
            self._data = (kraus_l, kraus_r)
            return self
        return Kraus((kraus_l, kraus_r), *self.dims)

    def transpose(self, inplace=False):
        """Return the transpose channel"""
        kraus_l, kraus_r = self._data
        kraus_l = [k.T for k in kraus_l]
        if kraus_r is not None:
            kraus_r = [k.T for k in kraus_r]
        dout, din = self.dims
        if inplace:
            self._data = (kraus_l, kraus_r)
            self._output_dim = dout
            self._input_dim = din
            return self
        return Kraus((kraus_l, kraus_r), din, dout)

    def compose(self, other, inplace=False, front=False):
        """Return Kraus representation for the composition channel B(A(input))

        Args:
            other (QuantumChannel): A channel rep object
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If True compose in reverse order A(B(input)) [default: False]

        Returns:
            Kraus: The Kraus representation for the composition channel.

        Raises:
            QiskitError: if dimensions do not allow channels to be composed.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        # Check dimensions match up
        if front and self._input_dim != other._output_dim:
            raise QiskitError('input_dim of self must match output_dim of other')
        if not front and self._output_dim != other._input_dim:
            raise QiskitError('input_dim of other must match output_dim of self')
        # Convert to Choi matrix
        if not isinstance(other, Kraus):
            other = Kraus(other)

        if front:
            ka_l, ka_r = self._data
            kb_l, kb_r = other._data
            input_dim = other._input_dim
            output_dim = self._output_dim
        else:
            ka_l, ka_r = other._data
            kb_l, kb_r = self._data
            input_dim = self._input_dim
            output_dim = other._output_dim

        kab_l = [np.dot(a, b) for a in ka_l for b in kb_l]
        if ka_r is None and kb_r is None:
            kab_r = None
        elif ka_r is None:
            kab_r = [np.dot(a, b) for a in ka_l for b in kb_r]
        elif kb_r is None:
            kab_r = [np.dot(a, b) for a in ka_r for b in kb_l]
        else:
            kab_r = [np.dot(a, b) for a in ka_r for b in kb_r]
        data = (kab_l, kab_r)
        if inplace:
            self._data = data
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        return Kraus(data, input_dim, output_dim)

    def tensor(self, other, inplace=False, front=False):
        """Return Kraus for the tensor product channel.

        Args:
            other (Kraus): A Kraus
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If False return (other ⊗ self),
                          if True return (self ⊗ other) [Default: False]
        Returns:
            Kraus: The Kraus for the composition channel.

        Raises:
            QiskitError: if b is not a Kraus object
        """
        # Convert other to Kraus
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        if not isinstance(other, Kraus):
            other = Kraus(other)

        # Get tensor matrix
        ka_l, ka_r = self._data
        kb_l, kb_r = other._data
        if front:
            kab_l = [np.kron(a, b) for a in ka_l for b in kb_l]
        else:
            kab_l = [np.kron(b, a) for a in ka_l for b in kb_l]
        if ka_r is None and kb_r is None:
            kab_r = None
        else:
            if ka_r is None:
                ka_r = ka_l
            if kb_r is None:
                kb_r = kb_l
            if front:
                kab_r = [np.kron(a, b) for a in ka_r for b in kb_r]
            else:
                kab_r = [np.kron(b, a) for a in ka_r for b in kb_r]
        data = (kab_l, kab_r)
        input_dim = self._input_dim * other._input_dim
        output_dim = self._output_dim * other._output_dim
        if inplace:
            self._data = data
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        return Kraus(data, input_dim, output_dim)

    def add(self, other, inplace=False):
        """Add another QuantumChannel"""
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        # Since we cannot directly add two channels in the Kraus
        # representation we try and use the other channels method
        # or convert to the Choi representation
        tmp = Kraus(Choi(self).add(other, inplace=True))
        if inplace:
            self._data = tmp._data
            self._input_dim = tmp._input_dim
            self._output_dim = tmp._output_dim
            return self
        return tmp

    def subtract(self, other, inplace=False):
        """Subtract another QuantumChannel"""
        # Since we cannot directly subtract two channels in the Kraus
        # representation we try and use the other channels method
        # or convert to the Choi representation
        tmp = Kraus(Choi(self).subtract(other, inplace=True))
        if inplace:
            self._data = tmp._data
            self._input_dim = tmp._input_dim
            self._output_dim = tmp._output_dim
            return self
        return tmp

    def multiply(self, other, inplace=False):
        """Multiple by a scalar"""
        if not isinstance(other, Number):
            raise QiskitError("Not a number")
        # If the number is complex we need to convert to general
        # kraus channel so we multiply via Choi representation
        if isinstance(other, complex) or other < 0:
            # Convert to Choi-matrix
            tmp = Kraus(Choi(self).multiply(other, inplace=True))
            if inplace:
                self._data = tmp._data
                self._input_dim = tmp._input_dim
                self._output_dim = tmp._output_dim
                return self
            return tmp
        # If the number is real we can update the Kraus operators
        # directly
        else:
            s = np.sqrt(other)
            if inplace:
                for j, _ in enumerate(self._data[0]):
                    self._data[0][j] *= s
                if self._data[1] is not None:
                    for j, _ in enumerate(self._data[1]):
                        self._data[1][j] *= s
            else:
                kraus_r = None
                kraus_l = [s * k for k in self._data[0]]
                if self._data[1] is not None:
                    kraus_r = [s * k for k in self._data[1]]
                return Kraus((kraus_l, kraus_r),
                             self._input_dim, self._output_dim)
