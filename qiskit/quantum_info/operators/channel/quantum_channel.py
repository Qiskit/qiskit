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
Abstract base class for Quantum Channels.
"""

from abc import abstractmethod
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _to_kraus
from qiskit.quantum_info.operators.channel.transformations import _to_operator


class QuantumChannel(BaseOperator):
    """Quantum channel representation base class."""

    def compose(self, other, qargs=None, front=False):
        """Return the composed quantum channel self @ other.

        Args:
            other (QuantumChannel): a quantum channel.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            QuantumChannel: The quantum channel self @ other.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or has
            incompatible dimensions.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            :class:`SuperOp` matrices. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        if front:
            return self._chanmul(other, qargs, left_multiply=False)
        return self._chanmul(other, qargs, left_multiply=True)

    def dot(self, other, qargs=None):
        """Return the right multiplied quantum channel self * other.

        Args:
            other (QuantumChannel): a quantum channel.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].

        Returns:
            QuantumChannel: The quantum channel self * other.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or has
            incompatible dimensions.
        """
        return super().dot(other, qargs=qargs)

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

    def to_instruction(self):
        """Convert to a Kraus or UnitaryGate circuit instruction.

        If the channel is unitary it will be added as a unitary gate,
        otherwise it will be added as a kraus simulator instruction.

        Returns:
            Instruction: A kraus instruction for the channel.

        Raises:
            QiskitError: if input data is not an N-qubit CPTP quantum channel.
        """
        from qiskit.circuit.instruction import Instruction
        # Check if input is an N-qubit CPTP channel.
        n_qubits = int(np.log2(self._input_dim))
        if self._input_dim != self._output_dim or 2**n_qubits != self._input_dim:
            raise QiskitError(
                'Cannot convert QuantumChannel to Instruction: channel is not an N-qubit channel.'
            )
        if not self.is_cptp():
            raise QiskitError(
                'Cannot convert QuantumChannel to Instruction: channel is not CPTP.'
            )
        # Next we convert to the Kraus representation. Since channel is CPTP we know
        # that there is only a single set of Kraus operators
        kraus, _ = _to_kraus(self.rep, self._data, *self.dim)
        # If we only have a single Kraus operator then the channel is
        # a unitary channel so can be converted to a UnitaryGate. We do this by
        # converting to an Operator and using its to_instruction method
        if len(kraus) == 1:
            return Operator(kraus[0]).to_instruction()
        return Instruction('kraus', n_qubits, 0, kraus)

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
                # flatten column-vector to vector
                state = np.reshape(state, shape[0])
        # Convert statevector to density matrix if required
        if density_matrix and ndim == 1:
            state = np.outer(state, np.transpose(np.conj(state)))
        return state

    @abstractmethod
    def _evolve(self, state, qargs=None):
        """Evolve a quantum state by the quantum channel.

        Args:
            state (DensityMatrix or Statevector): The input state.
            qargs (list): a list of quantum state subsystem positions to apply
                           the quantum channel on.

        Returns:
            DensityMatrix: the output quantum state as a density matrix.

        Raises:
            QiskitError: if the quantum channel dimension does not match the
            specified quantum state subsystem dimensions.
        """
        pass

    @abstractmethod
    def _chanmul(self, other, qargs=None, left_multiply=False):
        """Multiply two quantum channels.

        Args:
            other (QuantumChannel): a quantum channel.
            qargs (list): a list of subsystem positions to compose other on.
            left_multiply (bool): If True return other * self
                                  If False return self * other [Default:False]

        Returns:
            QuantumChannel: The composition channel.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        pass

    @classmethod
    def _init_transformer(cls, data):
        """Convert input into a QuantumChannel subclass object or Operator object"""
        # This handles common conversion for all QuantumChannel subclasses.
        # If the input is already a QuantumChannel subclass it will return
        # the original object
        if isinstance(data, QuantumChannel):
            return data
        if hasattr(data, 'to_quantumchannel'):
            # If the data object is not a QuantumChannel it will give
            # preference to a 'to_quantumchannel' attribute that allows
            # an arbitrary object to define its own conversion to any
            # quantum channel subclass.
            return data.to_quantumchannel()
        if hasattr(data, 'to_channel'):
            # TODO: this 'to_channel' method is the same case as the above
            # but is used by current version of Aer. It should be removed
            # once Aer is nupdated to use `to_quantumchannel`
            # instead of `to_channel`,
            return data.to_channel()
        # Finally if the input is not a QuantumChannel and doesn't have a
        # 'to_quantumchannel' conversion method we try and initialize it as a
        # regular matrix Operator which can be converted into a QuantumChannel.
        return Operator(data)
