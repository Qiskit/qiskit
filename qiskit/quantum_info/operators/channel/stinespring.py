# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_stinespring


class Stinespring(QuantumChannel):
    """Stinespring representation of a quantum channel"""

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize a quantum channel Stinespring operator.

        Args:
            data (QuantumCircuit or
                  Instruction or
                  BaseOperator or
                  matrix): data to initialize superoperator.
            input_dims (tuple): the input subsystem dimensions.
                                [Default: None]
            output_dims (tuple): the output subsystem dimensions.
                                 [Default: None]

        Raises:
            QiskitError: if input data cannot be initialized as a
            a list of Kraus matrices.

        Additional Information
        ----------------------
        If the input or output dimensions are None, they will be
        automatically determined from the input data. This can fail for the
        Stinespring operator if the output dimension cannot be automatically
        determined.
        """
        # If the input is a list or tuple we assume it is a pair of general
        # Stinespring matrices. If it is a numpy array we assume that it is
        # a single Stinespring matrix.
        if isinstance(data, (list, tuple, np.ndarray)):
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
            # If two Stinespring matrices check they are same shape
            if stine[1] is not None:
                if stine[1].shape != (dim_left, dim_right):
                    raise QiskitError("Invalid Stinespring input.")
            input_dim = dim_right
            if output_dims:
                output_dim = np.product(output_dims)
            else:
                output_dim = input_dim
            if dim_left % output_dim != 0:
                raise QiskitError("Invalid output_dim")
        else:
            # Otherwise we initialize by conversion from another Qiskit
            # object into the QuantumChannel.
            if isinstance(data, (QuantumCircuit, Instruction)):
                # If the input is a Terra QuantumCircuit or Instruction we
                # convert it to a SuperOp
                data = SuperOp._instruction_to_superop(data)
            else:
                # We use the QuantumChannel init transform to intialize
                # other objects into a QuantumChannel or Operator object.
                data = self._init_transformer(data)
            data = self._init_transformer(data)
            input_dim, output_dim = data.dim
            # Now that the input is an operator we convert it to a
            # Stinespring operator
            stine = _to_stinespring(data.rep, data._data, input_dim,
                                    output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()

        # Check and format input and output dimensions
        input_dims = self._automatic_dims(input_dims, input_dim)
        output_dims = self._automatic_dims(output_dims, output_dim)
        # Initialize either single or general Stinespring
        if stine[1] is None or (stine[1] == stine[0]).all():
            # Standard Stinespring map
            super().__init__(
                'Stinespring', (stine[0], None),
                input_dims=input_dims,
                output_dims=output_dims)
        else:
            # General (non-CPTP) Stinespring map
            super().__init__(
                'Stinespring',
                stine,
                input_dims=input_dims,
                output_dims=output_dims)

    @property
    def data(self):
        # Override to deal with data being either tuple or not
        if self._data[1] is None:
            return self._data[0]
        else:
            return self._data

    def is_cptp(self, atol=None, rtol=None):
        """Return True if completely-positive trace-preserving."""
        if atol is None:
            atol = self._atol
        if rtol is None:
            rtol = self._rtol
        if self._data[1] is not None:
            return False
        check = np.dot(np.transpose(np.conj(self._data[0])), self._data[0])
        return is_identity_matrix(check, rtol=self._rtol, atol=self._atol)

    def conjugate(self):
        """Return the conjugate of the QuantumChannel."""
        # pylint: disable=assignment-from-no-return
        stine_l = np.conjugate(self._data[0])
        stine_r = None
        if self._data[1] is not None:
            stine_r = np.conjugate(self._data[1])
        return Stinespring((stine_l, stine_r), self.input_dims(),
                           self.output_dims())

    def transpose(self):
        """Return the transpose of the QuantumChannel."""
        din, dout = self.dim
        dtr = self._data[0].shape[0] // dout
        stine = [None, None]
        for i, mat in enumerate(self._data):
            if mat is not None:
                stine[i] = np.reshape(
                    np.transpose(np.reshape(mat, (dout, dtr, din)), (2, 1, 0)),
                    (din * dtr, dout))
        return Stinespring(
            tuple(stine),
            input_dims=self.output_dims(),
            output_dims=self.input_dims())

    def compose(self, other, qargs=None, front=False):
        """Return the composition channel self∘other.

        Args:
            other (QuantumChannel): a quantum channel subclass.
            qargs (list): a list of subsystem positions to compose other on.
            front (bool): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            Stinespring: The composition channel as a Stinespring object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        if qargs is not None:
            return Stinespring(
                SuperOp(self).compose(other, qargs=qargs, front=front))

        # Convert other to Kraus
        if not isinstance(other, Kraus):
            other = Kraus(other)
        # Check dimensions match up
        if front and self._input_dim != other._output_dim:
            raise QiskitError(
                'input_dim of self must match output_dim of other')
        if not front and self._output_dim != other._input_dim:
            raise QiskitError(
                'input_dim of other must match output_dim of self')
        # Since we cannot directly compose two channels in Stinespring
        # representation we convert to the Kraus representation
        return Stinespring(Kraus(self).compose(other, front=front))

    def power(self, n):
        """The matrix power of the channel.

        Args:
            n (int): compute the matrix power of the superoperator matrix.

        Returns:
            Stinespring: the matrix power of the SuperOp converted to a
            Stinespring channel.

        Raises:
            QiskitError: if the input and output dimensions of the
            QuantumChannel are not equal, or the power is not an integer.
        """
        if n > 0:
            return super().power(n)
        return Stinespring(SuperOp(self).power(n))

    def tensor(self, other):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel subclass.

        Returns:
            Stinespring: the tensor product channel other ⊗ self as a
            Stinespring object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel subclass.

        Returns:
            Stinespring: the tensor product channel other ⊗ self as a
            Stinespring object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=True)

    def add(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (QuantumChannel): a quantum channel subclass.

        Returns:
            Stinespring: the linear addition self + other as a
            Stinespring object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        # Since we cannot directly add two channels in the Stinespring
        # representation we convert to the Choi representation
        return Stinespring(Choi(self).add(other))

    def subtract(self, other):
        """Return the QuantumChannel self - other.

        Args:
            other (QuantumChannel): a quantum channel subclass.

        Returns:
            Stinespring: the linear subtraction self - other as
            Stinespring object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        # Since we cannot directly subtract two channels in the Stinespring
        # representation we convert to the Choi representation
        return Stinespring(Choi(self).subtract(other))

    def multiply(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (complex): a complex number.

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
            return Stinespring(Choi(self).multiply(other))
        # If the number is real we can update the Kraus operators
        # directly
        num = np.sqrt(other)
        stine_l, stine_r = self._data
        stine_l = num * self._data[0]
        stine_r = None
        if self._data[1] is not None:
            stine_r = num * self._data[1]
        return Stinespring((stine_l, stine_r), self.input_dims(),
                           self.output_dims())

    def _evolve(self, state, qargs=None):
        """Evolve a quantum state by the QuantumChannel.

        Args:
            state (QuantumState): The input statevector or density matrix.
            qargs (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            QuantumState: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
            specified QuantumState subsystem dimensions.
        """
        # If subsystem evolution we use the SuperOp representation
        if qargs is not None:
            return SuperOp(self)._evolve(state, qargs)

        # Otherwise we compute full evolution directly
        state = self._format_state(state)
        if state.shape[0] != self._input_dim:
            raise QiskitError(
                "QuantumChannel input dimension is not equal to state dimension."
            )
        if state.ndim == 1 and self._data[1] is None and \
           self._data[0].shape[0] // self._output_dim == 1:
            # If the shape of the Stinespring operator is equal to the output_dim
            # evolution of a state vector psi -> stine.psi
            return np.dot(self._data[0], state)
        # Otherwise we always return a density matrix
        state = self._format_state(state, density_matrix=True)
        stine_l, stine_r = self._data
        if stine_r is None:
            stine_r = stine_l
        din, dout = self.dim
        dtr = stine_l.shape[0] // dout
        shape = (dout, dtr, din)
        return np.einsum('iAB,BC,jAC->ij', np.reshape(stine_l, shape), state,
                         np.reshape(np.conjugate(stine_r), shape))

    def _tensor_product(self, other, reverse=False):
        """Return the tensor product channel.

        Args:
            other (QuantumChannel): a quantum channel subclass.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False]
        Returns:
            Stinespring: the tensor product channel as a Stinespring object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        # Convert other to Stinespring
        if not isinstance(other, Stinespring):
            other = Stinespring(other)

        # Tensor Stinespring ops
        sa_l, sa_r = self._data
        sb_l, sb_r = other._data

        # Reshuffle tensor dimensions
        din_a, dout_a = self.dim
        din_b, dout_b = other.dim
        dtr_a = sa_l.shape[0] // dout_a
        dtr_b = sb_l.shape[0] // dout_b
        if reverse:
            shape_in = (dout_b, dtr_b, dout_a, dtr_a, din_b * din_a)
            shape_out = (dout_b * dtr_b * dout_a * dtr_a, din_b * din_a)
        else:
            shape_in = (dout_a, dtr_a, dout_b, dtr_b, din_a * din_b)
            shape_out = (dout_a * dtr_a * dout_b * dtr_b, din_a * din_b)

        # Compute left Stinespring op
        if reverse:
            input_dims = self.input_dims() + other.input_dims()
            output_dims = self.output_dims() + other.output_dims()
            sab_l = np.kron(sb_l, sa_l)
        else:
            input_dims = other.input_dims() + self.input_dims()
            output_dims = other.output_dims() + self.output_dims()
            sab_l = np.kron(sa_l, sb_l)
        # Reravel indices
        sab_l = np.reshape(
            np.transpose(np.reshape(sab_l, shape_in), (0, 2, 1, 3, 4)),
            shape_out)

        # Compute right Stinespring op
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
            # Reravel indices
            sab_r = np.reshape(
                np.transpose(np.reshape(sab_r, shape_in), (0, 2, 1, 3, 4)),
                shape_out)
        return Stinespring((sab_l, sab_r), input_dims, output_dims)
