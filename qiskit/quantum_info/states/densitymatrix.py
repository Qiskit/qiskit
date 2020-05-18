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
DensityMatrix quantum state class.
"""

import warnings
from numbers import Number
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.states.statevector import Statevector


class DensityMatrix(QuantumState):
    """DensityMatrix class"""

    def __init__(self, data, dims=None):
        """Initialize a density matrix object.

        Args:
            data (matrix_like or vector_like): a density matrix or
                statevector. If a vector the density matrix is constructed
                as the projector of that vector.
            dims (int or tuple or list): Optional. The subsystem dimension
                    of the state (See additional information).

        Raises:
            QiskitError: if input data is not valid.

        Additional Information:
            The ``dims`` kwarg can be None, an integer, or an iterable of
            integers.

            * ``Iterable`` -- the subsystem dimensions are the values in the list
              with the total number of subsystems given by the length of the list.

            * ``Int`` or ``None`` -- the leading dimension of the input matrix
              specifies the total dimension of the density matrix. If it is a
              power of two the state will be initialized as an N-qubit state.
              If it is not a power of two the state will have a single
              d-dimensional subsystem.
        """
        if isinstance(data, (list, np.ndarray)):
            # Finally we check if the input is a raw matrix in either a
            # python list or numpy array format.
            self._data = np.asarray(data, dtype=complex)
        elif hasattr(data, 'to_operator'):
            # If the data object has a 'to_operator' attribute this is given
            # higher preference than the 'to_matrix' method for initializing
            # an Operator object.
            op = data.to_operator()
            self._data = op.data
            if dims is None:
                dims = op._output_dims
        elif hasattr(data, 'to_matrix'):
            # If no 'to_operator' attribute exists we next look for a
            # 'to_matrix' attribute to a matrix that will be cast into
            # a complex numpy matrix.
            self._data = np.asarray(data.to_matrix(), dtype=complex)
        else:
            raise QiskitError("Invalid input data format for DensityMatrix")
        # Convert statevector into a density matrix
        ndim = self._data.ndim
        shape = self._data.shape
        if ndim == 2 and shape[0] == shape[1]:
            pass  # We good
        elif ndim == 1:
            self._data = np.outer(self._data, np.conj(self._data))
        elif ndim == 2 and shape[1] == 1:
            self._data = np.reshape(self._data, shape[0])
            shape = self._data.shape
        else:
            raise QiskitError(
                "Invalid DensityMatrix input: not a square matrix.")
        super().__init__(self._automatic_dims(dims, shape[0]))

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(
            self._data, other._data, rtol=self.rtol, atol=self.atol)

    def __repr__(self):
        prefix = 'DensityMatrix('
        pad = len(prefix) * ' '
        return '{}{},\n{}dims={})'.format(
            prefix, np.array2string(
                self._data, separator=', ', prefix=prefix),
            pad, self._dims)

    @property
    def data(self):
        """Return data."""
        return self._data

    def is_valid(self, atol=None, rtol=None):
        """Return True if trace 1 and positive semidefinite."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        # Check trace == 1
        if not np.allclose(self.trace(), 1, rtol=rtol, atol=atol):
            return False
        # Check Hermitian
        if not is_hermitian_matrix(self.data, rtol=rtol, atol=atol):
            return False
        # Check positive semidefinite
        return is_positive_semidefinite_matrix(self.data, rtol=rtol, atol=atol)

    def to_operator(self):
        """Convert to Operator"""
        dims = self.dims()
        return Operator(self.data, input_dims=dims, output_dims=dims)

    def conjugate(self):
        """Return the conjugate of the density matrix."""
        return DensityMatrix(np.conj(self.data), dims=self.dims())

    def trace(self):
        """Return the trace of the density matrix."""
        return np.trace(self.data)

    def purity(self):
        """Return the purity of the quantum state."""
        # For a valid statevector the purity is always 1, however if we simply
        # have an arbitrary vector (not correctly normalized) then the
        # purity is equivalent to the trace squared:
        # P(|psi>) = Tr[|psi><psi|psi><psi|] = |<psi|psi>|^2
        return np.trace(np.dot(self.data, self.data))

    def tensor(self, other):
        """Return the tensor product state self ⊗ other.

        Args:
            other (DensityMatrix): a quantum state object.

        Returns:
            DensityMatrix: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        dims = other.dims() + self.dims()
        data = np.kron(self._data, other._data)
        return DensityMatrix(data, dims)

    def expand(self, other):
        """Return the tensor product state other ⊗ self.

        Args:
            other (DensityMatrix): a quantum state object.

        Returns:
            DensityMatrix: the tensor product state other ⊗ self.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        dims = self.dims() + other.dims()
        data = np.kron(other._data, self._data)
        return DensityMatrix(data, dims)

    def _add(self, other):
        """Return the linear combination self + other.

        Args:
            other (DensityMatrix): a quantum state object.

        Returns:
            DensityMatrix: the linear combination self + other.

        Raises:
            QiskitError: if other is not a quantum state, or has
                         incompatible dimensions.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        if self.dim != other.dim:
            raise QiskitError("other DensityMatrix has different dimensions.")
        return DensityMatrix(self.data + other.data, self.dims())

    def _multiply(self, other):
        """Return the scalar multiplied state other * self.

        Args:
            other (complex): a complex number.

        Returns:
            DensityMatrix: the scalar multiplied state other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        return DensityMatrix(other * self.data, self.dims())

    def evolve(self, other, qargs=None):
        """Evolve a quantum state by an operator.

        Args:
            other (Operator or QuantumChannel
                   or Instruction or Circuit): The operator to evolve by.
            qargs (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            QuantumState: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
                         specified QuantumState subsystem dimensions.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        # Evolution by a circuit or instruction
        if isinstance(other, (QuantumCircuit, Instruction)):
            return self._evolve_instruction(other, qargs=qargs)

        # Evolution by a QuantumChannel
        if hasattr(other, 'to_quantumchannel'):
            return other.to_quantumchannel()._evolve(self, qargs=qargs)
        if isinstance(other, QuantumChannel):
            return other._evolve(self, qargs=qargs)

        # Unitary evolution by an Operator
        if not isinstance(other, Operator):
            other = Operator(other)
        return self._evolve_operator(other, qargs=qargs)

    def probabilities(self, qargs=None, decimals=None):
        """Return the subsystem measurement probability vector.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            np.array: The Numpy vector array of probabilities.

        Examples:

            Consider a 2-qubit product state :math:`\\rho=\\rho_1\\otimes\\rho_0`
            with :math:`\\rho_1=|+\\rangle\\!\\langle+|`,
            :math:`\\rho_0=|0\\rangle\\!\\langle0|`.

            .. jupyter-execute::

                from qiskit.quantum_info import DensityMatrix

                rho = DensityMatrix.from_label('+0')

                # Probabilities for measuring both qubits
                probs = rho.probabilities()
                print('probs: {}'.format(probs))

                # Probabilities for measuring only qubit-0
                probs_qubit_0 = rho.probabilities([0])
                print('Qubit-0 probs: {}'.format(probs_qubit_0))

                # Probabilities for measuring only qubit-1
                probs_qubit_1 = rho.probabilities([1])
                print('Qubit-1 probs: {}'.format(probs_qubit_1))

            We can also permute the order of qubits in the ``qargs`` list
            to change the qubit position in the probabilities output

            .. jupyter-execute::

                from qiskit.quantum_info import DensityMatrix

                rho = DensityMatrix.from_label('+0')

                # Probabilities for measuring both qubits
                probs = rho.probabilities([0, 1])
                print('probs: {}'.format(probs))

                # Probabilities for measuring both qubits
                # but swapping qubits 0 and 1 in output
                probs_swapped = rho.probabilities([1, 0])
                print('Swapped probs: {}'.format(probs_swapped))
        """
        probs = self._subsystem_probabilities(
            np.abs(self.data.diagonal()), self._dims, qargs=qargs)
        if decimals is not None:
            probs = probs.round(decimals=decimals)
        return probs

    def reset(self, qargs=None):
        """Reset state or subsystems to the 0-state.

        Args:
            qargs (list or None): subsystems to reset, if None all
                                  subsystems will be reset to their 0-state
                                  (Default: None).

        Returns:
            DensityMatrix: the reset state.

        Additional Information:
            If all subsystems are reset this will return the ground state
            on all subsystems. If only a some subsystems are reset this
            function will perform evolution by the reset
            :class:`~qiskit.quantum_info.SuperOp` of the reset subsystems.
        """
        if qargs is None:
            # Resetting all qubits does not require sampling or RNG
            state = np.zeros(2 * (self._dim, ), dtype=complex)
            state[0, 0] = 1
            return DensityMatrix(state, dims=self._dims)

        # Reset by evolving by reset SuperOp
        dims = self.dims(qargs)
        reset_superop = SuperOp(ScalarOp(dims, coeff=0))
        reset_superop.data[0] = Operator(ScalarOp(dims)).data.ravel()
        return self.evolve(reset_superop, qargs=qargs)

    @classmethod
    def from_label(cls, label):
        r"""Return a tensor product of Pauli X,Y,Z eigenstates.

        .. list-table:: Single-qubit state labels
           :header-rows: 1

           * - Label
             - Statevector
           * - ``"0"``
             - :math:`\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}`
           * - ``"1"``
             - :math:`\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}`
           * - ``"+"``
             - :math:`\frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}`
           * - ``"-"``
             - :math:`\frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}`
           * - ``"r"``
             - :math:`\frac{1}{2}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}`
           * - ``"l"``
             - :math:`\frac{1}{2}\begin{pmatrix} 1 & i \\ -i & 1 \end{pmatrix}`

        Args:
            label (string): a eigenstate string ket label (see table for
                            allowed values).

        Returns:
            Statevector: The N-qubit basis state density matrix.

        Raises:
            QiskitError: if the label contains invalid characters, or the length
                         of the label is larger than an explicitly specified num_qubits.
        """
        return DensityMatrix(Statevector.from_label(label))

    @staticmethod
    def from_int(i, dims):
        """Return a computational basis state density matrix.

        Args:
            i (int): the basis state element.
            dims (int or tuple or list): The subsystem dimensions of the statevector
                                         (See additional information).

        Returns:
            DensityMatrix: The computational basis state :math:`|i\\rangle\\!\\langle i|`.

        Additional Information:
            The ``dims`` kwarg can be an integer or an iterable of integers.

            * ``Iterable`` -- the subsystem dimensions are the values in the list
              with the total number of subsystems given by the length of the list.

            * ``Int`` -- the integer specifies the total dimension of the
              state. If it is a power of two the state will be initialized
              as an N-qubit state. If it is not a power of  two the state
              will have a single d-dimensional subsystem.
        """
        size = np.product(dims)
        state = np.zeros((size, size), dtype=complex)
        state[i, i] = 1.0
        return DensityMatrix(state, dims=dims)

    @classmethod
    def from_instruction(cls, instruction):
        """Return the output density matrix of an instruction.

        The statevector is initialized in the state :math:`|{0,\\ldots,0}\\rangle` of
        the same number of qubits as the input instruction or circuit, evolved
        by the input instruction, and the output statevector returned.

        Args:
            instruction (qiskit.circuit.Instruction or QuantumCircuit): instruction or circuit

        Returns:
            DensityMatrix: the final density matrix.

        Raises:
            QiskitError: if the instruction contains invalid instructions for
                         density matrix simulation.
        """
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an the statevector in the all |0> state
        num_qubits = instruction.num_qubits
        init = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        init[0, 0] = 1
        vec = DensityMatrix(init, dims=num_qubits * (2, ))
        vec._append_instruction(instruction)
        return vec

    def to_dict(self, decimals=None):
        r"""Convert the density matrix to dictionary form.

        This dictionary representation uses a Ket-like notation where the
        dictionary keys are qudit strings for the subsystem basis vectors.
        If any subsystem has a dimension greater than 10 comma delimiters are
        inserted between integers so that subsystems can be distinguished.

        Args:
            decimals (None or int): the number of decimal places to round
                                    values. If None no rounding is done
                                    (Default: None).

        Returns:
            dict: the dictionary form of the DensityMatrix.

        Examples:

            The ket-form of a 2-qubit density matrix
            :math:`rho = |-\rangle\!\langle -|\otimes |0\rangle\!\langle 0|`

            .. jupyter-execute::

                from qiskit.quantum_info import DensityMatrix

                rho = DensityMatrix.from_label('-0')
                print(rho.to_dict())

            For non-qubit subsystems the integer range can go from 0 to 9. For
            example in a qutrit system

            .. jupyter-execute::

                import numpy as np
                from qiskit.quantum_info import DensityMatrix

                mat = np.zeros((9, 9))
                mat[0, 0] = 0.25
                mat[3, 3] = 0.25
                mat[6, 6] = 0.25
                mat[-1, -1] = 0.25
                rho = DensityMatrix(mat, dims=(3, 3))
                print(rho.to_dict())

            For large subsystem dimensions delimeters are required. The
            following example is for a 20-dimensional system consisting of
            a qubit and 10-dimensional qudit.

            .. jupyter-execute::

                import numpy as np
                from qiskit.quantum_info import DensityMatrix

                mat = np.zeros((2 * 10, 2 * 10))
                mat[0, 0] = 0.5
                mat[-1, -1] = 0.5
                rho = DensityMatrix(mat, dims=(2, 10))
                print(rho.to_dict())
        """
        return self._matrix_to_dict(self.data,
                                    self._dims,
                                    decimals=decimals,
                                    string_labels=True)

    @property
    def _shape(self):
        """Return the tensor shape of the matrix operator"""
        return 2 * tuple(reversed(self.dims()))

    def _evolve_operator(self, other, qargs=None):
        """Evolve density matrix by an operator"""
        if qargs is None:
            # Evolution on full matrix
            if self._dim != other._input_dim:
                raise QiskitError(
                    "Operator input dimension is not equal to density matrix dimension."
                )
            op_mat = other.data
            mat = np.dot(op_mat, self.data).dot(op_mat.T.conj())
            return DensityMatrix(mat, dims=other._output_dims)
        # Otherwise we are applying an operator only to subsystems
        # Check dimensions of subsystems match the operator
        if self.dims(qargs) != other.input_dims():
            raise QiskitError(
                "Operator input dimensions are not equal to statevector subsystem dimensions."
            )
        # Reshape statevector and operator
        tensor = np.reshape(self.data, self._shape)
        # Construct list of tensor indices of statevector to be contracted
        num_indices = len(self.dims())
        indices = [num_indices - 1 - qubit for qubit in qargs]
        # Left multiple by mat
        mat = np.reshape(other.data, other._shape)
        tensor = Operator._einsum_matmul(tensor, mat, indices)
        # Right multiply by mat ** dagger
        adj = other.adjoint()
        mat_adj = np.reshape(adj.data, adj._shape)
        tensor = Operator._einsum_matmul(tensor, mat_adj, indices, num_indices,
                                         True)
        # Replace evolved dimensions
        new_dims = list(self.dims())
        for i, qubit in enumerate(qargs):
            new_dims[qubit] = other._output_dims[i]
        new_dim = np.product(new_dims)
        return DensityMatrix(np.reshape(tensor, (new_dim, new_dim)),
                             dims=new_dims)

    def _append_instruction(self, other, qargs=None):
        """Update the current Statevector by applying an instruction."""
        from qiskit.circuit.reset import Reset
        from qiskit.circuit.barrier import Barrier

        # Try evolving by a matrix operator (unitary-like evolution)
        mat = Operator._instruction_to_matrix(other)
        if mat is not None:
            self._data = self._evolve_operator(Operator(mat), qargs=qargs).data
            return

        # Special instruction types
        if isinstance(other, Reset):
            self._data = self.reset(qargs)._data
            return
        if isinstance(other, Barrier):
            return

        # Otherwise try evolving by a Superoperator
        chan = SuperOp._instruction_to_superop(other)
        if chan is not None:
            # Evolve current state by the superoperator
            self._data = chan._evolve(self, qargs=qargs).data
            return
        # If the instruction doesn't have a matrix defined we use its
        # circuit decomposition definition if it exists, otherwise we
        # cannot compose this gate and raise an error.
        if other.definition is None:
            raise QiskitError('Cannot apply Instruction: {}'.format(
                other.name))
        for instr, qregs, cregs in other.definition:
            if cregs:
                raise QiskitError(
                    'Cannot apply instruction with classical registers: {}'.
                    format(instr.name))
            # Get the integer position of the flat register
            if qargs is None:
                new_qargs = [tup.index for tup in qregs]
            else:
                new_qargs = [qargs[tup.index] for tup in qregs]
            self._append_instruction(instr, qargs=new_qargs)

    def _evolve_instruction(self, obj, qargs=None):
        """Return a new statevector by applying an instruction."""
        if isinstance(obj, QuantumCircuit):
            obj = obj.to_instruction()
        vec = DensityMatrix(self.data, dims=self._dims)
        vec._append_instruction(obj, qargs=qargs)
        return vec

    def to_statevector(self, atol=None, rtol=None):
        """Return a statevector from a pure density matrix.

        Args:
            atol (float): Absolute tolerance for checking operation validity.
            rtol (float): Relative tolerance for checking operation validity.

        Returns:
            Statevector: The pure density matrix's corresponding statevector.
                Corresponds to the eigenvector of the only non-zero eigenvalue.

        Raises:
            QiskitError: if the state is not pure.
        """
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        if not is_hermitian_matrix(self._data, atol=atol, rtol=rtol):
            raise QiskitError("Not a valid density matrix (non-hermitian).")

        evals, evecs = np.linalg.eig(self._data)

        nonzero_evals = evals[abs(evals) > atol]
        if len(nonzero_evals) != 1 or not np.isclose(nonzero_evals[0], 1,
                                                     atol=atol, rtol=rtol):
            raise QiskitError("Density matrix is not a pure state")

        psi = evecs[:, np.argmax(evals)]  # eigenvectors returned in columns.
        return Statevector(psi)

    def to_counts(self):
        """Returns the density matrix as a counts dict of probabilities.

        DEPRECATED: use :meth:`probabilities_dict` instead.

        Returns:
            dict: Counts of probabilities.
        """
        warnings.warn(
            'The `Statevector.to_counts` method is deprecated as of 0.13.0,'
            ' and will be removed no earlier than 3 months after that '
            'release date. You should use the `Statevector.probabilities_dict`'
            ' method instead.', DeprecationWarning, stacklevel=2)
        return self.probabilities_dict()
