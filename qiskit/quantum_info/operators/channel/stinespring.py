# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Stinespring representation of a Quantum Channel.


The Stinespring representation for a quantum channel E is given by a rectangular matrix A such that

    E(ρ) = Tr_2[A.ρ.A^dagger]

A general operator map G can also be written using the generalized Kraus representation which
is given by two matrices A, B such that

    G(ρ) = Tr_2[A.ρ.B^dagger]

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
from .kraus import Kraus
from .choi import Choi
from .transformations import _to_stinespring


class Stinespring(QuantumChannel):
    """Stinespring representation of a quantum channel"""

    def __init__(self, data, input_dim=None, output_dim=None):
        # Check if input is a quantum channel object
        # If so we disregard the dimension kwargs
        if issubclass(data.__class__, QuantumChannel):
            input_dim, output_dim = data.dims
            stine = _to_stinespring(data.rep, data._data, input_dim,
                                    output_dim)
        else:
            if not isinstance(data, tuple):
                # Convert single Stinespring set to length 1 tuple
                stine = (np.array(data, dtype=complex), None)
            if isinstance(data, tuple) and len(data) == 2:
                if data[1] is None:
                    stine = (np.array(data[0], dtype=complex), None)
                else:
                    stine = (np.array(data[0], dtype=complex),
                             np.array(data[1], dtype=complex))

            dim_left, dim_right = stine[0].shape
            # If two stinespring matrices check they are same shape
            if stine[1] is not None:
                if stine[1].shape != (dim_left, dim_right):
                    raise QiskitError("Invalid Stinespring input.")
            if input_dim is None:
                input_dim = dim_right
            if output_dim is None:
                output_dim = input_dim
            if input_dim != dim_right:
                raise QiskitError("Invalid input dimension")
            if dim_left % output_dim != 0:
                raise QiskitError("Invalid output dimension")
        # Record the dimension to be traced over for stinespring
        # evolution
        if stine[1] is None or (stine[1] == stine[0]).all():
            # Standard Stinespring map
            super().__init__(
                'Stinespring', (stine[0], None),
                input_dim=input_dim,
                output_dim=output_dim)
        else:
            # General (non-CPTP) Stinespring map
            super().__init__(
                'Stinespring',
                stine,
                input_dim=input_dim,
                output_dim=output_dim)

    @property
    def data(self):
        # Override to deal with data being either tuple or not
        if self._data[1] is None:
            return self._data[0]
        else:
            return self._data

    def _evolve(self, state):
        """Evolve a quantum state by the QuantumChannel.

        Args:
            state (QuantumState): The input statevector or density matrix.

        Returns:
            QuantumState: the output quantum state.
        """
        state = self._check_state(state)
        if state.ndim == 1 and self._data[1] is None and \
           self._data[0].shape[0] // self._output_dim == 1:
            # If the shape of the stinespring operator is equal to the output_dim
            # evolution of a state vector psi -> stine.psi
            return np.dot(self._data[0], state)
        # Otherwise we always return a density matrix
        state = self._format_density_matrix(state)
        state = self._format_density_matrix(self._check_state(state))
        stine_l, stine_r = self._data
        if stine_r is None:
            stine_r = stine_l
        din, dout = self.dims
        dtr = stine_l.shape[0] // dout
        shape = (dout, dtr, din)
        return np.einsum('iAB,BC,jAC->ij', np.reshape(stine_l, shape), state,
                         np.reshape(np.conjugate(stine_r), shape))

    def is_cptp(self):
        """Return True if completely-positive trace-preserving."""
        # We convert to the Choi representation to check if CPTP
        if self._data[1] is not None:
            return False
        check = np.dot(np.transpose(np.conj(self._data[0])), self._data[0])
        return is_identity_matrix(check, rtol=self.rtol, atol=self.atol)

    def conjugate(self, inplace=False):
        """Return the conjugate of the  QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Stinespring: the conjugate of the quantum channel as a
            Stinespring object.
        """
        # pylint: disable=assignment-from-no-return
        if inplace:
            np.conjugate(self._data[0], out=self._data[0])
            if self._data[1] is not None:
                np.conjugate(self._data[1], out=self._data[1])
            return self
        stine_l = np.conjugate(self._data[0])
        stine_r = None
        if self._data[1] is not None:
            stine_r = np.conjugate(self._data[1])
        return Stinespring((stine_l, stine_r), *self.dims)

    def transpose(self, inplace=False):
        """Return the transpose of the QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Stinespring: the transpose of the quantum channel as a
            Stinespring object.
        """
        din, dout = self.dims
        dtr = self._data[0].shape[0] // dout
        stine = [None, None]
        for i, mat in enumerate(self._data):
            if mat is not None:
                stine[i] = np.reshape(
                    np.transpose(np.reshape(mat, (dout, dtr, din)), (2, 1, 0)),
                    (din * dtr, dout))
        if inplace:
            self._data = tuple(stine)
            self._input_dim = dout
            self._output_dim = din
            return self
        # Return new stinespring operator with output and input dims swapped
        return Stinespring(tuple(stine), input_dim=dout, output_dim=din)

    def adjoint(self, inplace=False):
        """Return the adjoint of the QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Stinespring: the adjoint of the quantum channel as a
            Stinespring object.
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
            Stinespring: The composition channel as a Stinespring object.

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
        # Since we cannot directly compose two channels in Stinespring
        # representation we convert to the Kraus representation
        tmp = Stinespring(
            Kraus(self).compose(other, inplace=True, front=front))
        if inplace:
            self._data = tmp._data
            self._input_dim = tmp._input_dim
            self._output_dim = tmp._output_dim
            return self
        return tmp

    def power(self, n, inplace=False):
        """Return the compose of a QuantumChannel with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).
            inplace (bool): If True modify the current object inplace
                            [Default: False]

        Returns:
            Stinespring: the n-times composition channel as a Stinespring
            object.

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
            Stinespring: the tensor product channel other ⊗ self as a
            Stinespring object.

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
            Stinespring: the tensor product channel other ⊗ self as a
            Stinespring object.

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
            Stinespring: the linear addition self + other as a
            Stinespring object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        # Since we cannot directly add two channels in the Stinespring
        # representation we convert to the Choi representation
        tmp = Stinespring(Choi(self).add(other, inplace=True))
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
            Stinespring: the linear subtraction self - other as
            Stinespring object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        # Since we cannot directly subtract two channels in the Stinespring
        # representation we convert to the Choi representation
        tmp = Stinespring(Choi(self).subtract(other, inplace=True))
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
            Stinespring: the scalar multiplication other * self as a
            Stinespring object.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        # If the number is complex or negative we need to convert to
        # general Stinespring representation so we first convert to
        # the Choi representation
        if isinstance(other, complex) or other < 1:
            # Convert to Choi-matrix
            tmp = Stinespring(Choi(self).multiply(other, inplace=True))
            if inplace:
                self._data = tmp._data
                self._input_dim = tmp._input_dim
                self._output_dim = tmp._output_dim
                return self
            return tmp
        # If the number is real we can update the Kraus operators
        # directly
        num = np.sqrt(other)
        if inplace:
            self._data[0] *= num
            if self._data[1] is not None:
                self._data[1] *= num
            return self
        stine_l, stine_r = self._data
        stine_l = num * self._data[0]
        stine_r = None
        if self._data[1] is not None:
            stine_r = num * self._data[1]
        return Stinespring((stine_l, stine_r), *self.dims)

    def _tensor_product(self, other, inplace=False, reverse=False):
        """Return the tensor product channel.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                            [default: False]
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False]
        Returns:
            Stinespring: the tensor product channel as a Stinespring object.

        Raises:
            QiskitError: if other is not a Stinespring object
        """
        # Convert other to Stinespring
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
        if not isinstance(other, Stinespring):
            other = Stinespring(other)

        # Tensor stinespring ops
        sa_l, sa_r = self._data
        sb_l, sb_r = other._data

        # Reshuffle tensor dimensions
        din_a, dout_a = self.dims
        din_b, dout_b = other.dims
        dtr_a = sa_l.shape[0] // dout_a
        dtr_b = sb_l.shape[0] // dout_b
        if reverse:
            shape_in = (dout_b, dtr_b, dout_a, dtr_a, din_b * din_a)
            shape_out = (dout_b * dtr_b * dout_a * dtr_a, din_b * din_a)
        else:
            shape_in = (dout_a, dtr_a, dout_b, dtr_b, din_a * din_b)
            shape_out = (dout_a * dtr_a * dout_b * dtr_b, din_a * din_b)

        # Compute left stinepsring op
        if reverse:
            sab_l = np.kron(sb_l, sa_l)
        else:
            sab_l = np.kron(sa_l, sb_l)
        # Reravel indicies
        sab_l = np.reshape(
            np.transpose(np.reshape(sab_l, shape_in), (0, 2, 1, 3, 4)),
            shape_out)

        # Compute right stinespring op
        if sa_r is None and sb_r is None:
            sab_r = None
        else:
            if sa_r is None:
                sa_r = sa_l
            elif sb_r is None:
                sb_r = sb_l
            if reverse:
                sab_r = np.kron(sb_r, sa_r)
            else:
                sab_r = np.kron(sa_r, sb_r)
            # Reravel indicies
            sab_r = np.reshape(
                np.transpose(np.reshape(sab_r, shape_in), (0, 2, 1, 3, 4)),
                shape_out)
        if inplace:
            self._data = (sab_l, sab_r)
            self._input_dim = din_a * din_b
            self._output_dim = dout_a * dout_b
            return self
        return Stinespring((sab_l, sab_r), din_a * din_b, dout_a * dout_b)
