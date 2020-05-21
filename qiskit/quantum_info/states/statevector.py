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
Statevector quantum state class.
"""

import re
import warnings
from numbers import Number

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal


class Statevector(QuantumState):
    """Statevector class"""

    def __init__(self, data, dims=None):
        """Initialize a statevector object.

        Args:
            data (vector_like): a complex statevector.
            dims (int or tuple or list): Optional. The subsystem dimension of
                                         the state (See additional information).

        Raises:
            QiskitError: if input data is not valid.

        Additional Information:
            The ``dims`` kwarg can be None, an integer, or an iterable of
            integers.

            * ``Iterable`` -- the subsystem dimensions are the values in the list
              with the total number of subsystems given by the length of the list.

            * ``Int`` or ``None`` -- the length of the input vector
              specifies the total dimension of the density matrix. If it is a
              power of two the state will be initialized as an N-qubit state.
              If it is not a power of two the state will have a single
              d-dimensional subsystem.
        """
        if isinstance(data, (list, np.ndarray)):
            # Finally we check if the input is a raw vector in either a
            # python list or numpy array format.
            self._data = np.asarray(data, dtype=complex)
        elif isinstance(data, Statevector):
            self._data = data._data
            if dims is None:
                dims = data._dims
        elif isinstance(data, Operator):
            # We allow conversion of column-vector operators to Statevectors
            input_dim, output_dim = data.dim
            if input_dim != 1:
                raise QiskitError("Input Operator is not a column-vector.")
            self._data = np.ravel(data.data)
        else:
            raise QiskitError("Invalid input data format for Statevector")
        # Check that the input is a numpy vector or column-vector numpy
        # matrix. If it is a column-vector matrix reshape to a vector.
        ndim = self._data.ndim
        shape = self._data.shape
        if ndim != 1:
            if ndim == 2 and shape[1] == 1:
                self._data = np.reshape(self._data, shape[0])
            elif ndim != 2 or shape[1] != 1:
                raise QiskitError("Invalid input: not a vector or column-vector.")
        super().__init__(self._automatic_dims(dims, shape[0]))

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(
            self._data, other._data, rtol=self.rtol, atol=self.atol)

    def __repr__(self):
        prefix = 'Statevector('
        pad = len(prefix) * ' '
        return '{}{},\n{}dims={})'.format(
            prefix, np.array2string(
                self.data, separator=', ', prefix=prefix),
            pad, self._dims)

    @property
    def data(self):
        """Return data."""
        return self._data

    def is_valid(self, atol=None, rtol=None):
        """Return True if a Statevector has norm 1."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        norm = np.linalg.norm(self.data)
        return np.allclose(norm, 1, rtol=rtol, atol=atol)

    def to_operator(self):
        """Convert state to a rank-1 projector operator"""
        mat = np.outer(self.data, np.conj(self.data))
        return Operator(mat, input_dims=self.dims(), output_dims=self.dims())

    def conjugate(self):
        """Return the conjugate of the operator."""
        return Statevector(np.conj(self.data), dims=self.dims())

    def trace(self):
        """Return the trace of the quantum state as a density matrix."""
        return np.sum(np.abs(self.data) ** 2)

    def purity(self):
        """Return the purity of the quantum state."""
        # For a valid statevector the purity is always 1, however if we simply
        # have an arbitrary vector (not correctly normalized) then the
        # purity is equivalent to the trace squared:
        # P(|psi>) = Tr[|psi><psi|psi><psi|] = |<psi|psi>|^2
        return self.trace() ** 2

    def tensor(self, other):
        """Return the tensor product state self ⊗ other.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            Statevector: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        dims = other.dims() + self.dims()
        data = np.kron(self._data, other._data)
        return Statevector(data, dims)

    def expand(self, other):
        """Return the tensor product state other ⊗ self.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            Statevector: the tensor product state other ⊗ self.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        dims = self.dims() + other.dims()
        data = np.kron(other._data, self._data)
        return Statevector(data, dims)

    def _add(self, other):
        """Return the linear combination self + other.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            Statevector: the linear combination self + other.

        Raises:
            QiskitError: if other is not a quantum state, or has
                         incompatible dimensions.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        if self.dim != other.dim:
            raise QiskitError("other Statevector has different dimensions.")
        return Statevector(self.data + other.data, self.dims())

    def _multiply(self, other):
        """Return the scalar multiplied state self * other.

        Args:
            other (complex): a complex number.

        Returns:
            Statevector: the scalar multiplied state other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        return Statevector(other * self.data, self.dims())

    def evolve(self, other, qargs=None):
        """Evolve a quantum state by the operator.

        Args:
            other (Operator): The operator to evolve by.
            qargs (list): a list of Statevector subsystem positions to apply
                           the operator on.

        Returns:
            Statevector: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
                         specified Statevector subsystem dimensions.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        # Evolution by a circuit or instruction
        if isinstance(other, (QuantumCircuit, Instruction)):
            return self._evolve_instruction(other, qargs=qargs)
        # Evolution by an Operator
        if not isinstance(other, Operator):
            other = Operator(other)
        if qargs is None:
            # Evolution on full statevector
            if self._dim != other._input_dim:
                raise QiskitError(
                    "Operator input dimension is not equal to statevector dimension."
                )
            return Statevector(np.dot(other.data, self.data), dims=other.output_dims())
        # Otherwise we are applying an operator only to subsystems
        # Check dimensions of subsystems match the operator
        if self.dims(qargs) != other.input_dims():
            raise QiskitError(
                "Operator input dimensions are not equal to statevector subsystem dimensions."
            )
        # Reshape statevector and operator
        tensor = np.reshape(self.data, self._shape)
        mat = np.reshape(other.data, other._shape)
        # Construct list of tensor indices of statevector to be contracted
        num_indices = len(self.dims())
        indices = [num_indices - 1 - qubit for qubit in qargs]
        tensor = Operator._einsum_matmul(tensor, mat, indices)
        new_dims = list(self.dims())
        for i, qubit in enumerate(qargs):
            new_dims[qubit] = other._output_dims[i]
        # Replace evolved dimensions
        return Statevector(np.reshape(tensor, np.product(new_dims)), dims=new_dims)

    def equiv(self, other, rtol=None, atol=None):
        """Return True if statevectors are equivalent up to global phase.

        Args:
            other (Statevector): a statevector object.
            rtol (float): relative tolerance value for comparison.
            atol (float): absolute tolerance value for comparison.

        Returns:
            bool: True if statevectors are equivalent up to global phase.
        """
        if not isinstance(other, Statevector):
            try:
                other = Statevector(other)
            except QiskitError:
                return False
        if self.dim != other.dim:
            return False
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return matrix_equal(self.data, other.data, ignore_phase=True,
                            rtol=rtol, atol=atol)

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

            Consider a 2-qubit product state
            :math:`|\\psi\\rangle=|+\\rangle\\otimes|0\\rangle`.

            .. jupyter-execute::

                from qiskit.quantum_info import Statevector

                psi = Statevector.from_label('+0')

                # Probabilities for measuring both qubits
                probs = psi.probabilities()
                print('probs: {}'.format(probs))

                # Probabilities for measuring only qubit-0
                probs_qubit_0 = psi.probabilities([0])
                print('Qubit-0 probs: {}'.format(probs_qubit_0))

                # Probabilities for measuring only qubit-1
                probs_qubit_1 = psi.probabilities([1])
                print('Qubit-1 probs: {}'.format(probs_qubit_1))

            We can also permute the order of qubits in the ``qargs`` list
            to change the qubit position in the probabilities output

            .. jupyter-execute::

                from qiskit.quantum_info import Statevector

                psi = Statevector.from_label('+0')

                # Probabilities for measuring both qubits
                probs = psi.probabilities([0, 1])
                print('probs: {}'.format(probs))

                # Probabilities for measuring both qubits
                # but swapping qubits 0 and 1 in output
                probs_swapped = psi.probabilities([1, 0])
                print('Swapped probs: {}'.format(probs_swapped))
        """
        probs = self._subsystem_probabilities(
            np.abs(self.data) ** 2, self._dims, qargs=qargs)
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
            Statevector: the reset state.

        Additional Information:
            If all subsystems are reset this will return the ground state
            on all subsystems. If only a some subsystems are reset this
            function will perform a measurement on those subsystems and
            evolve the subsystems so that the collapsed post-measurement
            states are rotated to the 0-state. The RNG seed for this
            sampling can be set using the :meth:`seed` method.
        """
        if qargs is None:
            # Resetting all qubits does not require sampling or RNG
            state = np.zeros(self._dim, dtype=complex)
            state[0] = 1
            return Statevector(state, dims=self._dims)

        # Sample a single measurement outcome
        dims = self.dims(qargs)
        probs = self.probabilities(qargs)
        sample = self._rng.choice(len(probs), p=probs, size=1)

        # Convert to projector for state update
        proj = np.zeros(len(probs), dtype=complex)
        proj[sample] = 1 / np.sqrt(probs[sample])

        # Rotate outcome to 0
        reset = np.eye(len(probs))
        reset[0, 0] = 0
        reset[sample, sample] = 0
        reset[0, sample] = 1

        # compose with reset projection
        reset = np.dot(reset, np.diag(proj))
        return self.evolve(
            Operator(reset, input_dims=dims, output_dims=dims),
            qargs=qargs)

    def to_counts(self):
        """Returns the statevector as a counts dict
        of probabilities.

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

    @classmethod
    def from_label(cls, label):
        """Return a tensor product of Pauli X,Y,Z eigenstates.

        .. list-table:: Single-qubit state labels
           :header-rows: 1

           * - Label
             - Statevector
           * - ``"0"``
             - :math:`[1, 0]`
           * - ``"1"``
             - :math:`[0, 1]`
           * - ``"+"``
             - :math:`[1 / \\sqrt{2},  1 / \\sqrt{2}]`
           * - ``"-"``
             - :math:`[1 / \\sqrt{2},  -1 / \\sqrt{2}]`
           * - ``"r"``
             - :math:`[1 / \\sqrt{2},  i / \\sqrt{2}]`
           * - ``"l"``
             - :math:`[1 / \\sqrt{2},  -i / \\sqrt{2}]`

        Args:
            label (string): a eigenstate string ket label (see table for
                            allowed values).

        Returns:
            Statevector: The N-qubit basis state density matrix.

        Raises:
            QiskitError: if the label contains invalid characters, or the
                         length of the label is larger than an explicitly
                         specified num_qubits.
        """
        # Check label is valid
        if re.match(r'^[01rl\-+]+$', label) is None:
            raise QiskitError('Label contains invalid characters.')
        # We can prepare Z-eigenstates by converting the computational
        # basis bit-string to an integer and preparing that unit vector
        # However, for X-basis states, we will prepare a Z-eigenstate first
        # then apply Hadamard gates to rotate 0 and 1s to + and -.
        z_label = label
        xy_states = False
        if re.match('^[01]+$', label) is None:
            # We have X or Y eigenstates so replace +,r with 0 and
            # -,l with 1 and prepare the corresponding Z state
            xy_states = True
            z_label = z_label.replace('+', '0')
            z_label = z_label.replace('r', '0')
            z_label = z_label.replace('-', '1')
            z_label = z_label.replace('l', '1')
        # Initialize Z eigenstate vector
        num_qubits = len(label)
        data = np.zeros(1 << num_qubits, dtype=complex)
        pos = int(z_label, 2)
        data[pos] = 1
        state = Statevector(data)
        if xy_states:
            # Apply hadamards to all qubits in X eigenstates
            x_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            # Apply S.H to qubits in Y eigenstates
            y_mat = np.dot(np.diag([1, 1j]), x_mat)
            for qubit, char in enumerate(reversed(label)):
                if char in ['+', '-']:
                    state = state.evolve(x_mat, qargs=[qubit])
                elif char in ['r', 'l']:
                    state = state.evolve(y_mat, qargs=[qubit])
        return state

    @staticmethod
    def from_int(i, dims):
        """Return a computational basis statevector.

        Args:
            i (int): the basis state element.
            dims (int or tuple or list): The subsystem dimensions of the statevector
                                         (See additional information).

        Returns:
            Statevector: The computational basis state :math:`|i\\rangle`.

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
        state = np.zeros(size, dtype=complex)
        state[i] = 1.0
        return Statevector(state, dims=dims)

    @classmethod
    def from_instruction(cls, instruction):
        """Return the output statevector of an instruction.

        The statevector is initialized in the state :math:`|{0,\\ldots,0}\\rangle` of the
        same number of qubits as the input instruction or circuit, evolved
        by the input instruction, and the output statevector returned.

        Args:
            instruction (qiskit.circuit.Instruction or QuantumCircuit): instruction or circuit

        Returns:
            Statevector: The final statevector.

        Raises:
            QiskitError: if the instruction contains invalid instructions for
                         the statevector simulation.
        """
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an the statevector in the all |0> state
        init = np.zeros(2 ** instruction.num_qubits, dtype=complex)
        init[0] = 1.0
        vec = Statevector(init, dims=instruction.num_qubits * (2,))
        vec._append_instruction(instruction)
        return vec

    def to_dict(self, decimals=None):
        r"""Convert the statevector to dictionary form.

        This dictionary representation uses a Ket-like notation where the
        dictionary keys are qudit strings for the subsystem basis vectors.
        If any subsystem has a dimension greater than 10 comma delimiters are
        inserted between integers so that subsystems can be distinguished.

        Args:
            decimals (None or int): the number of decimal places to round
                                    values. If None no rounding is done
                                    (Default: None).

        Returns:
            dict: the dictionary form of the Statevector.

        Example:

            The ket-form of a 2-qubit statevector
            :math:`|\psi\rangle = |-\rangle\otimes |0\rangle`

            .. jupyter-execute::

                from qiskit.quantum_info import Statevector

                psi = Statevector.from_label('-0')
                print(psi.to_dict())

            For non-qubit subsystems the integer range can go from 0 to 9. For
            example in a qutrit system

            .. jupyter-execute::

                import numpy as np
                from qiskit.quantum_info import Statevector

                vec = np.zeros(9)
                vec[0] = 1 / np.sqrt(2)
                vec[-1] = 1 / np.sqrt(2)
                psi = Statevector(vec, dims=(3, 3))
                print(psi.to_dict())

            For large subsystem dimensions delimeters are required. The
            following example is for a 20-dimensional system consisting of
            a qubit and 10-dimensional qudit.

            .. jupyter-execute::

                import numpy as np
                from qiskit.quantum_info import Statevector

                vec = np.zeros(2 * 10)
                vec[0] = 1 / np.sqrt(2)
                vec[-1] = 1 / np.sqrt(2)
                psi = Statevector(vec, dims=(2, 10))
                print(psi.to_dict())
        """
        return self._vector_to_dict(self.data,
                                    self._dims,
                                    decimals=decimals,
                                    string_labels=True)

    @property
    def _shape(self):
        """Return the tensor shape of the matrix operator"""
        return tuple(reversed(self.dims()))

    def _append_instruction(self, obj, qargs=None):
        """Update the current Statevector by applying an instruction."""
        from qiskit.circuit.reset import Reset
        from qiskit.circuit.barrier import Barrier

        mat = Operator._instruction_to_matrix(obj)
        if mat is not None:
            # Perform the composition and inplace update the current state
            # of the operator
            self._data = self.evolve(mat, qargs=qargs).data
            return

        # Special instruction types
        if isinstance(obj, Reset):
            self._data = self.reset(qargs)._data
            return
        if isinstance(obj, Barrier):
            return

        # If the instruction doesn't have a matrix defined we use its
        # circuit decomposition definition if it exists, otherwise we
        # cannot compose this gate and raise an error.
        if obj.definition is None:
            raise QiskitError('Cannot apply Instruction: {}'.format(obj.name))

        for instr, qregs, cregs in obj.definition:
            if cregs:
                raise QiskitError(
                    'Cannot apply instruction with classical registers: {}'.format(
                        instr.name))
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
        vec = Statevector(self.data, dims=self.dims())
        vec._append_instruction(obj, qargs=qargs)
        return vec
