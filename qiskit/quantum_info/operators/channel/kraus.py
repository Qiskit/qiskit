# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=len-as-condition
"""
Kraus representation of a Quantum Channel.


The Kraus representation for a quantum channel E is given by a set of matrics [A_i] such that

    E(ρ) = sum_i A_i.ρ.A_i^dagger

A general operator map G can also be written using the generalized Kraus representation which
is given by two sets of matrices [A_i], [B_i] such that

    G(ρ) = sum_i A_i.ρ.B_i^dagger

See [1] for further details.

References:
    [1] C.J. Wood, J.D. Biamonte, D.G. Cory, Quant. Inf. Comp. 15, 0579-0811 (2015)
        Open access: arXiv:1111.6950 [quant-ph]
"""

from numbers import Number

import numpy as np

from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.predicates import is_identity_matrix
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
                for i in data[1:]:
                    op = np.array(i, dtype=complex)
                    if op.shape != shape:
                        raise QiskitError(
                            "Kraus operators are different dimensions.")
                    kraus.append(op)
                # Convert single Kraus set to general Kraus pair
                kraus = (kraus, None)

            # Check if generalized Kraus set ([A_i], [B_i]) for channel:
            # E(rho) = sum_i A_i * rho * B_i^dagger
            elif isinstance(data,
                            tuple) and len(data) == 2 and len(data[0]) > 0:
                kraus_left = [np.array(data[0][0], dtype=complex)]
                shape = kraus_left[0].shape
                for i in data[0][1:]:
                    op = np.array(i, dtype=complex)
                    if op.shape != shape:
                        raise QiskitError(
                            "Kraus operators are different dimensions.")
                    kraus_left.append(op)
                if data[1] is None:
                    kraus = (kraus_left, None)
                else:
                    kraus_right = []
                    for i in data[1]:
                        op = np.array(i, dtype=complex)
                        if op.shape != shape:
                            raise QiskitError(
                                "Kraus operators are different dimensions.")
                        kraus_right.append(op)
                    kraus = (kraus_left, kraus_right)
            else:
                raise QiskitError("Invalid input for Kraus channel.")
        dout, din = kraus[0][0].shape
        if (input_dim and input_dim != din) or (output_dim
                                                and output_dim != dout):
            raise QiskitError("Invalid dimensions for Kraus input.")

        if kraus[1] is None or np.allclose(kraus[0], kraus[1]):
            # Standard Kraus map
            super().__init__(
                'Kraus', (kraus[0], None), input_dim=din, output_dim=dout)
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

    def is_cptp(self):
        """Return True if completely-positive trace-preserving."""
        if self._data[1] is not None:
            return False
        accum = 0j
        for op in self._data[0]:
            accum += np.dot(np.transpose(np.conj(op)), op)
        return is_identity_matrix(accum, rtol=self.rtol, atol=self.atol)

    def _evolve(self, state):
        """Evolve a quantum state by the QuantumChannel.

        Args:
            state (QuantumState): The input statevector or density matrix.

        Returns:
            QuantumState: the output quantum state.
        """
        state = self._check_state(state)
        if state.ndim == 1 and self._data[1] is None and len(
                self._data[0]) == 1:
            # If we only have a single Kraus operator we can implement unitary-type
            # evolution of a state vector psi -> K[0].psi
            return np.dot(self._data[0][0], state)
        # Otherwise we always return a density matrix
        state = self._format_density_matrix(state)
        kraus_l, kraus_r = self._data
        if kraus_r is None:
            kraus_r = kraus_l
        return np.einsum('AiB,BC,AjC->ij', kraus_l, state,
                         np.conjugate(kraus_r))

    def canonical(self, inplace=False):
        """Convert to canonical Kraus representation.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Kraus: the canonical Kraus decomposition.
        """
        tmp = Kraus(Choi(self))
        if inplace:
            self._data = tmp._data
            return self
        return tmp

    def conjugate(self, inplace=False):
        """Return the conjugate of the  QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Kraus: the conjugate of the quantum channel as a Kraus object.
        """
        kraus_l, kraus_r = self._data
        kraus_l = [k.conj() for k in kraus_l]
        if kraus_r is not None:
            kraus_r = [k.conj() for k in kraus_r]
        if inplace:
            self._data = (kraus_l, kraus_r)
            return self
        return Kraus((kraus_l, kraus_r), *self.dims)

    def transpose(self, inplace=False):
        """Return the transpose of the QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Kraus: the transpose of the quantum channel as a Kraus object.
        """
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

    def adjoint(self, inplace=False):
        """Return the adjoint of the QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Kraus: the adjoint of the quantum channel as a Kraus object.
        """
        return super().adjoint(inplace=inplace)

    def compose(self, other, inplace=False, front=False):
        """Return the composition channel self∘other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                            [Default: False]
            front (bool): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            Kraus: The composition channel as a Kraus object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
        # Check dimensions match up
        if front and self._input_dim != other._output_dim:
            raise QiskitError(
                'input_dim of self must match output_dim of other')
        if not front and self._output_dim != other._input_dim:
            raise QiskitError(
                'input_dim of other must match output_dim of self')
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

    def power(self, n, inplace=False):
        """Return the compose of a QuantumChannel with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).
            inplace (bool): If True modify the current object inplace
                            [Default: False]

        Returns:
            Kraus: the n-times composition channel as a Kraus object.

        Raises:
            QiskitError: if the input and output dimensions of the
            QuantumChannel are not equal, or the power is not a positive
            integer.
        """
        return super().power(n, inplace=inplace)

    def tensor(self, other, inplace=False):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Kraus: the tensor product channel self ⊗ other as a Kraus
            object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass.
        """
        return self._tensor_product(other, inplace=inplace, reverse=False)

    def expand(self, other, inplace=False):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Kraus: the tensor product channel other ⊗ self as a Kraus
            object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass.
        """
        return self._tensor_product(other, inplace=inplace, reverse=True)

    def add(self, other, inplace=False):
        """Return the QuantumChannel self + other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Kraus: the linear addition self + other as a Kraus object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
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
        """Return the QuantumChannel self - other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Kraus: the linear subtraction self - other as Kraus object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
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
        """Return the QuantumChannel self + other.

        Args:
            other (complex): a complex number
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Kraus: the scalar multiplication other * self as a Kraus object.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
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
        val = np.sqrt(other)
        if inplace:
            for j, _ in enumerate(self._data[0]):
                self._data[0][j] *= val
            if self._data[1] is not None:
                for j, _ in enumerate(self._data[1]):
                    self._data[1][j] *= val
            return self
        kraus_r = None
        kraus_l = [val * k for k in self._data[0]]
        if self._data[1] is not None:
            kraus_r = [val * k for k in self._data[1]]
        return Kraus((kraus_l, kraus_r), self._input_dim, self._output_dim)

    def _tensor_product(self, other, inplace=False, reverse=False):
        """Return the tensor product channel.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                            [default: False]
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            Kraus: the tensor product channel as a Kraus object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass.
        """
        # Convert other to Kraus
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
        if not isinstance(other, Kraus):
            other = Kraus(other)

        # Get tensor matrix
        ka_l, ka_r = self._data
        kb_l, kb_r = other._data
        if reverse:
            kab_l = [np.kron(b, a) for a in ka_l for b in kb_l]
        else:
            kab_l = [np.kron(a, b) for a in ka_l for b in kb_l]
        if ka_r is None and kb_r is None:
            kab_r = None
        else:
            if ka_r is None:
                ka_r = ka_l
            if kb_r is None:
                kb_r = kb_l
            if reverse:
                kab_r = [np.kron(b, a) for a in ka_r for b in kb_r]
            else:
                kab_r = [np.kron(a, b) for a in ka_r for b in kb_r]
        data = (kab_l, kab_r)
        input_dim = self._input_dim * other._input_dim
        output_dim = self._output_dim * other._output_dim
        if inplace:
            self._data = data
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        return Kraus(data, input_dim, output_dim)
