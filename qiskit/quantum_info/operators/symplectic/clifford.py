# This code is part of Qiskit.
#
# (C) Copyright IBM 2017--2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Clifford operator class.
"""
from __future__ import annotations

import functools
import itertools
import re
from typing import Literal

import numpy as np

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, IGate, SGate, XGate, YGate, ZGate
from qiskit.circuit.operation import Operation
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import _count_y
from qiskit.utils.deprecation import deprecate_func
from qiskit.synthesis.linear import calc_inverse_matrix

from .base_pauli import BasePauli
from .clifford_circuits import _append_circuit, _append_operation


class Clifford(BaseOperator, AdjointMixin, Operation):
    """An N-qubit unitary operator from the Clifford group.

    **Representation**

    An *N*-qubit Clifford operator is stored as a length *2N × (2N+1)*
    boolean tableau using the convention from reference [1].

    * Rows 0 to *N-1* are the *destabilizer* group generators
    * Rows *N* to *2N-1* are the *stabilizer* group generators.

    The internal boolean tableau for the Clifford
    can be accessed using the :attr:`tableau` attribute. The destabilizer or
    stabilizer rows can each be accessed as a length-N Stabilizer table using
    :attr:`destab` and :attr:`stab` attributes.

    A more easily human readable representation of the Clifford operator can
    be obtained by calling the :meth:`to_dict` method. This representation is
    also used if a Clifford object is printed as in the following example

    .. code-block::

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Clifford

        # Bell state generation circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        cliff = Clifford(qc)

        # Print the Clifford
        print(cliff)

        # Print the Clifford destabilizer rows
        print(cliff.to_labels(mode="D"))

        # Print the Clifford stabilizer rows
        print(cliff.to_labels(mode="S"))

    .. parsed-literal::

        Clifford: Stabilizer = ['+XX', '+ZZ'], Destabilizer = ['+IZ', '+XI']
        ['+IZ', '+XI']
        ['+XX', '+ZZ']

    **Circuit Conversion**

    Clifford operators can be initialized from circuits containing *only* the
    following Clifford gates: :class:`~qiskit.circuit.library.IGate`,
    :class:`~qiskit.circuit.library.XGate`, :class:`~qiskit.circuit.library.YGate`,
    :class:`~qiskit.circuit.library.ZGate`, :class:`~qiskit.circuit.library.HGate`,
    :class:`~qiskit.circuit.library.SGate`, :class:`~qiskit.circuit.library.SdgGate`,
    :class:`~qiskit.circuit.library.SXGate`, :class:`~qiskit.circuit.library.SXdgGate`,
    :class:`~qiskit.circuit.library.CXGate`, :class:`~qiskit.circuit.library.CZGate`,
    :class:`~qiskit.circuit.library.CYGate`, :class:`~qiskit.circuit.library.DXGate`,
    :class:`~qiskit.circuit.library.SwapGate`, :class:`~qiskit.circuit.library.iSwapGate`,
    :class:`~qiskit.circuit.library.ECRGate`, :class:`~qiskit.circuit.library.LinearFunction`,
    :class:`~qiskit.circuit.library.PermutationGate`.
    They can be converted back into a :class:`~qiskit.circuit.QuantumCircuit`,
    or :class:`~qiskit.circuit.Gate` object using the :meth:`~Clifford.to_circuit`
    or :meth:`~Clifford.to_instruction` methods respectively. Note that this
    decomposition is not necessarily optimal in terms of number of gates.

    .. note::

        A minimally generating set of gates for Clifford circuits is
        the :class:`~qiskit.circuit.library.HGate` and
        :class:`~qiskit.circuit.library.SGate` gate and *either* the
        :class:`~qiskit.circuit.library.CXGate` or
        :class:`~qiskit.circuit.library.CZGate` two-qubit gate.

    Clifford operators can also be converted to
    :class:`~qiskit.quantum_info.Operator` objects using the
    :meth:`to_operator` method. This is done via decomposing to a circuit, and then
    simulating the circuit as a unitary operator.

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """

    _COMPOSE_PHASE_LOOKUP = None
    _COMPOSE_1Q_LOOKUP = None

    def __array__(self, dtype=None):
        if dtype:
            return np.asarray(self.to_matrix(), dtype=dtype)
        return self.to_matrix()

    def __init__(self, data, validate=True, copy=True):
        """Initialize an operator object."""

        # pylint: disable=cyclic-import
        from qiskit.circuit.library import LinearFunction, PermutationGate

        # Initialize from another Clifford
        if isinstance(data, Clifford):
            num_qubits = data.num_qubits
            self.tableau = data.tableau.copy() if copy else data.tableau

        # Initialize from ScalarOp as N-qubit identity discarding any global phase
        elif isinstance(data, ScalarOp):
            if not data.num_qubits or not data.is_unitary():
                raise QiskitError("Can only initialize from N-qubit identity ScalarOp.")
            num_qubits = data.num_qubits
            self.tableau = np.fromfunction(
                lambda i, j: i == j, (2 * num_qubits, 2 * num_qubits + 1)
            ).astype(bool)

        # Initialize from LinearFunction
        elif isinstance(data, LinearFunction):
            num_qubits = len(data.linear)
            self.tableau = self.from_linear_function(data)

        # Initialize from PermutationGate
        elif isinstance(data, PermutationGate):
            num_qubits = len(data.pattern)
            self.tableau = self.from_permutation(data)

        # Initialize from a QuantumCircuit or Instruction object
        elif isinstance(data, (QuantumCircuit, Instruction)):
            num_qubits = data.num_qubits
            self.tableau = Clifford.from_circuit(data).tableau

        # Initialize StabilizerTable directly from the data
        else:
            if isinstance(data, (list, np.ndarray)) and np.asarray(data, dtype=bool).ndim == 2:
                data = np.array(data, dtype=bool, copy=copy)
                if data.shape[0] == data.shape[1]:
                    self.tableau = self._stack_table_phase(
                        data, np.zeros(data.shape[0], dtype=bool)
                    )
                    num_qubits = data.shape[0] // 2
                elif data.shape[0] + 1 == data.shape[1]:
                    self.tableau = data
                    num_qubits = data.shape[0] // 2
                else:
                    raise QiskitError("")
            else:
                n_paulis = len(data)
                symp = self._from_label(data[0])
                num_qubits = len(symp) // 2
                tableau = np.zeros((n_paulis, len(symp)), dtype=bool)
                tableau[0] = symp
                for i in range(1, n_paulis):
                    tableau[i] = self._from_label(data[i])
                self.tableau = tableau

            # Validate table is a symplectic matrix
            if validate and not Clifford._is_symplectic(self.symplectic_matrix):
                raise QiskitError(
                    "Invalid Clifford. Input StabilizerTable is not a valid symplectic matrix."
                )

        # Initialize BaseOperator
        super().__init__(num_qubits=num_qubits)

    @property
    def name(self):
        """Unique string identifier for operation type."""
        return "clifford"

    @property
    def num_clbits(self):
        """Number of classical bits."""
        return 0

    def __repr__(self):
        return f"Clifford({repr(self.tableau)})"

    def __str__(self):
        return (
            f'Clifford: Stabilizer = {self.to_labels(mode="S")}, '
            f'Destabilizer = {self.to_labels(mode="D")}'
        )

    def __eq__(self, other):
        """Check if two Clifford tables are equal"""
        return super().__eq__(other) and (self.tableau == other.tableau).all()

    def copy(self):
        return type(self)(self, validate=False, copy=True)

    # ---------------------------------------------------------------------
    # Attributes
    # ---------------------------------------------------------------------

    # pylint: disable=bad-docstring-quotes

    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="Instead, index or iterate through the Clifford.tableau attribute.",
    )
    def __getitem__(self, key):
        """Return a stabilizer Pauli row"""
        return self.table.__getitem__(key)

    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="Use Clifford.tableau property instead.",
    )
    def __setitem__(self, key, value):
        """Set a stabilizer Pauli row"""
        self.tableau.__setitem__(key, self._stack_table_phase(value.array, value.phase))

    @property
    def symplectic_matrix(self):
        """Return boolean symplectic matrix."""
        return self.tableau[:, :-1]

    @symplectic_matrix.setter
    def symplectic_matrix(self, value):
        self.tableau[:, :-1] = value

    @property
    def phase(self):
        """Return phase with boolean representation."""
        return self.tableau[:, -1]

    @phase.setter
    def phase(self, value):
        self.tableau[:, -1] = value

    @property
    def x(self):
        """The x array for the symplectic representation."""
        return self.tableau[:, 0 : self.num_qubits]

    @x.setter
    def x(self, value):
        self.tableau[:, 0 : self.num_qubits] = value

    @property
    def z(self):
        """The z array for the symplectic representation."""
        return self.tableau[:, self.num_qubits : 2 * self.num_qubits]

    @z.setter
    def z(self, value):
        self.tableau[:, self.num_qubits : 2 * self.num_qubits] = value

    @property
    def destab(self):
        """The destabilizer array for the symplectic representation."""
        return self.tableau[: self.num_qubits, :]

    @destab.setter
    def destab(self, value):
        self.tableau[: self.num_qubits, :] = value

    @property
    def destab_x(self):
        """The destabilizer x array for the symplectic representation."""
        return self.tableau[: self.num_qubits, : self.num_qubits]

    @destab_x.setter
    def destab_x(self, value):
        self.tableau[: self.num_qubits, : self.num_qubits] = value

    @property
    def destab_z(self):
        """The destabilizer z array for the symplectic representation."""
        return self.tableau[: self.num_qubits, self.num_qubits : 2 * self.num_qubits]

    @destab_z.setter
    def destab_z(self, value):
        self.tableau[: self.num_qubits, self.num_qubits : 2 * self.num_qubits] = value

    @property
    def destab_phase(self):
        """Return phase of destabilizer with boolean representation."""
        return self.tableau[: self.num_qubits, -1]

    @destab_phase.setter
    def destab_phase(self, value):
        self.tableau[: self.num_qubits, -1] = value

    @property
    def stab(self):
        """The stabilizer array for the symplectic representation."""
        return self.tableau[self.num_qubits :, :]

    @stab.setter
    def stab(self, value):
        self.tableau[self.num_qubits :, :] = value

    @property
    def stab_x(self):
        """The stabilizer x array for the symplectic representation."""
        return self.tableau[self.num_qubits :, : self.num_qubits]

    @stab_x.setter
    def stab_x(self, value):
        self.tableau[self.num_qubits :, : self.num_qubits] = value

    @property
    def stab_z(self):
        """The stabilizer array for the symplectic representation."""
        return self.tableau[self.num_qubits :, self.num_qubits : 2 * self.num_qubits]

    @stab_z.setter
    def stab_z(self, value):
        self.tableau[self.num_qubits :, self.num_qubits : 2 * self.num_qubits] = value

    @property
    def stab_phase(self):
        """Return phase of stabilizer with boolean representation."""
        return self.tableau[self.num_qubits :, -1]

    @stab_phase.setter
    def stab_phase(self, value):
        self.tableau[self.num_qubits :, -1] = value

    # ---------------------------------------------------------------------
    # Utility Operator methods
    # ---------------------------------------------------------------------

    def is_unitary(self):
        """Return True if the Clifford table is valid."""
        # A valid Clifford is always unitary, so this function is really
        # checking that the underlying Stabilizer table array is a valid
        # Clifford array.
        return Clifford._is_symplectic(self.symplectic_matrix)

    # ---------------------------------------------------------------------
    # BaseOperator Abstract Methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        return Clifford._conjugate_transpose(self, "C")

    def adjoint(self):
        return Clifford._conjugate_transpose(self, "A")

    def transpose(self):
        return Clifford._conjugate_transpose(self, "T")

    def tensor(self, other: Clifford) -> Clifford:
        if not isinstance(other, Clifford):
            other = Clifford(other)
        return self._tensor(self, other)

    def expand(self, other: Clifford) -> Clifford:
        if not isinstance(other, Clifford):
            other = Clifford(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        n = a.num_qubits + b.num_qubits
        tableau = np.zeros((2 * n, 2 * n + 1), dtype=bool)
        clifford = cls(tableau, validate=False)
        clifford.destab_x[: b.num_qubits, : b.num_qubits] = b.destab_x
        clifford.destab_x[b.num_qubits :, b.num_qubits :] = a.destab_x
        clifford.destab_z[: b.num_qubits, : b.num_qubits] = b.destab_z
        clifford.destab_z[b.num_qubits :, b.num_qubits :] = a.destab_z
        clifford.stab_x[: b.num_qubits, : b.num_qubits] = b.stab_x
        clifford.stab_x[b.num_qubits :, b.num_qubits :] = a.stab_x
        clifford.stab_z[: b.num_qubits, : b.num_qubits] = b.stab_z
        clifford.stab_z[b.num_qubits :, b.num_qubits :] = a.stab_z
        clifford.phase[: b.num_qubits] = b.destab_phase
        clifford.phase[b.num_qubits : n] = a.destab_phase
        clifford.phase[n : n + b.num_qubits] = b.stab_phase
        clifford.phase[n + b.num_qubits :] = a.stab_phase
        return clifford

    def compose(
        self,
        other: Clifford | QuantumCircuit | Instruction,
        qargs: list | None = None,
        front: bool = False,
    ) -> Clifford:
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        # If other is a QuantumCircuit we can more efficiently compose
        # using the _append_circuit method to update each gate recursively
        # to the current Clifford, rather than converting to a Clifford first
        # and then doing the composition of tables.
        if not front:
            if isinstance(other, QuantumCircuit):
                return _append_circuit(self.copy(), other, qargs=qargs)
            if isinstance(other, Instruction):
                return _append_operation(self.copy(), other, qargs=qargs)

        if not isinstance(other, Clifford):
            # Not copying is safe since we're going to drop our only reference to `other` at the end
            # of the function.
            other = Clifford(other, copy=False)

        # Validate compose dimensions
        self._op_shape.compose(other._op_shape, qargs, front)

        # Pad other with identities if composing on subsystem
        other = self._pad_with_identity(other, qargs)

        left, right = (self, other) if front else (other, self)

        if self.num_qubits == 1:
            return self._compose_1q(left, right)
        return self._compose_general(left, right)

    @classmethod
    def _compose_general(cls, first, second):
        # Correcting for phase due to Pauli multiplication. Start with factors of -i from XZ = -iY
        # on individual qubits, and then handle multiplication between each qubitwise pair.
        ifacts = np.sum(second.x & second.z, axis=1, dtype=int)

        x1, z1 = first.x.astype(np.uint8), first.z.astype(np.uint8)
        lookup = cls._compose_lookup()

        # The loop is over 2*n_qubits entries, and the entire loop is cubic in the number of qubits.
        for k, row2 in enumerate(second.symplectic_matrix):
            x1_select = x1[row2]
            z1_select = z1[row2]
            x1_accum = np.logical_xor.accumulate(x1_select, axis=0).astype(np.uint8)
            z1_accum = np.logical_xor.accumulate(z1_select, axis=0).astype(np.uint8)
            indexer = (x1_select[1:], z1_select[1:], x1_accum[:-1], z1_accum[:-1])
            ifacts[k] += np.sum(lookup[indexer])
        p = np.mod(ifacts, 4) // 2

        phase = (
            (np.matmul(second.symplectic_matrix, first.phase, dtype=int) + second.phase + p) % 2
        ).astype(bool)
        data = cls._stack_table_phase(
            (np.matmul(second.symplectic_matrix, first.symplectic_matrix, dtype=int) % 2).astype(
                bool
            ),
            phase,
        )
        return Clifford(data, validate=False, copy=False)

    @classmethod
    def _compose_1q(cls, first, second):
        # 1-qubit composition can be done with a simple lookup table; there are 24 elements in the
        # 1q Clifford group, so 576 possible combinations, which is small enough to look up.
        if cls._COMPOSE_1Q_LOOKUP is None:
            # The valid tables for 1q Cliffords.
            tables_1q = np.array(
                [
                    [[False, True], [True, False]],
                    [[False, True], [True, True]],
                    [[True, False], [False, True]],
                    [[True, False], [True, True]],
                    [[True, True], [False, True]],
                    [[True, True], [True, False]],
                ]
            )
            phases_1q = np.array([[False, False], [False, True], [True, False], [True, True]])
            # Build the lookup table.
            cliffords = [
                cls(cls._stack_table_phase(table, phase), validate=False, copy=False)
                for table, phase in itertools.product(tables_1q, phases_1q)
            ]
            cls._COMPOSE_1Q_LOOKUP = {
                (cls._hash(left), cls._hash(right)): cls._compose_general(left, right)
                for left, right in itertools.product(cliffords, repeat=2)
            }
        return cls._COMPOSE_1Q_LOOKUP[cls._hash(first), cls._hash(second)].copy()

    @classmethod
    def _compose_lookup(
        cls,
    ):
        if cls._COMPOSE_PHASE_LOOKUP is None:
            # A lookup table for calculating phases.  The indices are
            #     current_x, current_z, running_x_count, running_z_count
            # where all counts taken modulo 2.
            lookup = np.zeros((2, 2, 2, 2), dtype=int)
            lookup[0, 1, 1, 0] = lookup[1, 0, 1, 1] = lookup[1, 1, 0, 1] = -1
            lookup[0, 1, 1, 1] = lookup[1, 0, 0, 1] = lookup[1, 1, 1, 0] = 1
            lookup.setflags(write=False)
            cls._COMPOSE_PHASE_LOOKUP = lookup
        return cls._COMPOSE_PHASE_LOOKUP

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    def to_dict(self):
        """Return dictionary representation of Clifford object."""
        return {
            "stabilizer": self.to_labels(mode="S"),
            "destabilizer": self.to_labels(mode="D"),
        }

    @classmethod
    def from_dict(cls, obj):
        """Load a Clifford from a dictionary"""
        labels = obj.get("destabilizer") + obj.get("stabilizer")
        n_paulis = len(labels)
        symp = cls._from_label(labels[0])
        tableau = np.zeros((n_paulis, len(symp)), dtype=bool)
        tableau[0] = symp
        for i in range(1, n_paulis):
            tableau[i] = cls._from_label(labels[i])
        return cls(tableau)

    def to_matrix(self):
        """Convert operator to Numpy matrix."""
        return self.to_operator().data

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Clifford:
        """Create a Clifford from a unitary matrix.

        Note that this function takes exponentially long time w.r.t. the number of qubits.

        Args:
            matrix (np.array): A unitary matrix representing a Clifford to be converted.

        Returns:
            Clifford: the Clifford object for the unitary matrix.

        Raises:
            QiskitError: if the input is not a Clifford matrix.
        """
        tableau = cls._unitary_matrix_to_tableau(matrix)
        if tableau is None:
            raise QiskitError("Non-Clifford matrix is not convertible")
        return cls(tableau)

    @classmethod
    def from_linear_function(cls, linear_function):
        """Create a Clifford from a Linear Function.

        If the linear function is represented by a nxn binary invertible matrix A,
        then the corresponding Clifford has symplectic matrix [[A^t, 0], [0, A^{-1}]].

        Args:
            linear_function (LinearFunction): A linear function to be converted.

        Returns:
            Clifford: the Clifford object for this linear function.
        """

        mat = linear_function.linear
        mat_t = np.transpose(mat)
        mat_i = calc_inverse_matrix(mat)

        dim = len(mat)
        zero = np.zeros((dim, dim), dtype=int)
        symplectic_mat = np.block([[mat_t, zero], [zero, mat_i]])
        phase = np.zeros(2 * dim, dtype=int)
        tableau = cls._stack_table_phase(symplectic_mat, phase)
        return tableau

    @classmethod
    def from_permutation(cls, permutation_gate):
        """Create a Clifford from a PermutationGate.

        Args:
            permutation_gate (PermutationGate): A permutation to be converted.

        Returns:
            Clifford: the Clifford object for this permutation.
        """

        pat = permutation_gate.pattern
        dim = len(pat)
        symplectic_mat = np.zeros((2 * dim, 2 * dim), dtype=int)
        for i, j in enumerate(pat):
            symplectic_mat[j, i] = True
            symplectic_mat[j + dim, i + dim] = True
        phase = np.zeros(2 * dim, dtype=bool)
        tableau = cls._stack_table_phase(symplectic_mat, phase)
        return tableau

    def to_operator(self) -> Operator:
        """Convert to an Operator object."""
        return Operator(self.to_instruction())

    @classmethod
    def from_operator(cls, operator: Operator) -> Clifford:
        """Create a Clifford from a operator.

        Note that this function takes exponentially long time w.r.t. the number of qubits.

        Args:
            operator (Operator): An operator representing a Clifford to be converted.

        Returns:
            Clifford: the Clifford object for the operator.

        Raises:
            QiskitError: if the input is not a Clifford operator.
        """
        tableau = cls._unitary_matrix_to_tableau(operator.to_matrix())
        if tableau is None:
            raise QiskitError("Non-Clifford operator is not convertible")
        return cls(tableau)

    def to_circuit(self):
        """Return a QuantumCircuit implementing the Clifford.

        For N <= 3 qubits this is based on optimal CX cost decomposition
        from reference [1]. For N > 3 qubits this is done using the general
        non-optimal compilation routine from reference [2].

        Return:
            QuantumCircuit: a circuit implementation of the Clifford.

        References:
            1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
               structure of the Clifford group*,
               `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_

            2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
               Phys. Rev. A 70, 052328 (2004).
               `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
        """
        from qiskit.synthesis.clifford import synth_clifford_full

        return synth_clifford_full(self)

    def to_instruction(self):
        """Return a Gate instruction implementing the Clifford."""
        return self.to_circuit().to_gate()

    @staticmethod
    def from_circuit(circuit: QuantumCircuit | Instruction) -> Clifford:
        """Initialize from a QuantumCircuit or Instruction.

        Args:
            circuit (QuantumCircuit or ~qiskit.circuit.Instruction):
                instruction to initialize.

        Returns:
            Clifford: the Clifford object for the instruction.

        Raises:
            QiskitError: if the input instruction is non-Clifford or contains
                         classical register instruction.
        """
        if not isinstance(circuit, (QuantumCircuit, Instruction)):
            raise QiskitError("Input must be a QuantumCircuit or Instruction")

        # Initialize an identity Clifford
        clifford = Clifford(np.eye(2 * circuit.num_qubits), validate=False)
        if isinstance(circuit, QuantumCircuit):
            clifford = _append_circuit(clifford, circuit)
        else:
            clifford = _append_operation(clifford, circuit)
        return clifford

    @staticmethod
    def from_label(label: str) -> Clifford:
        """Return a tensor product of single-qubit Clifford gates.

        Args:
            label (string): single-qubit operator string.

        Returns:
            Clifford: The N-qubit Clifford operator.

        Raises:
            QiskitError: if the label contains invalid characters.

        Additional Information:
            The labels correspond to the single-qubit Cliffords are

            * - Label
              - Stabilizer
              - Destabilizer
            * - ``"I"``
              - +Z
              - +X
            * - ``"X"``
              - -Z
              - +X
            * - ``"Y"``
              - -Z
              - -X
            * - ``"Z"``
              - +Z
              - -X
            * - ``"H"``
              - +X
              - +Z
            * - ``"S"``
              - +Z
              - +Y
        """
        # Check label is valid
        label_gates = {
            "I": IGate(),
            "X": XGate(),
            "Y": YGate(),
            "Z": ZGate(),
            "H": HGate(),
            "S": SGate(),
        }
        if re.match(r"^[IXYZHS\-+]+$", label) is None:
            raise QiskitError("Label contains invalid characters.")
        # Initialize an identity matrix and apply each gate
        num_qubits = len(label)
        op = Clifford(np.eye(2 * num_qubits, dtype=bool))
        for qubit, char in enumerate(reversed(label)):
            op = _append_operation(op, label_gates[char], qargs=[qubit])
        return op

    def to_labels(self, array: bool = False, mode: Literal["S", "D", "B"] = "B"):
        r"""Convert a Clifford to a list Pauli (de)stabilizer string labels.

        For large Clifford converting using the ``array=True``
        kwarg will be more efficient since it allocates memory for
        the full Numpy array of labels in advance.

        .. list-table:: Stabilizer Representations
            :header-rows: 1

            * - Label
              - Phase
              - Symplectic
              - Matrix
              - Pauli
            * - ``"+I"``
              - 0
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
              - :math:`I`
            * - ``"-I"``
              - 1
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}`
              - :math:`-I`
            * - ``"X"``
              - 0
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
              - :math:`X`
            * - ``"-X"``
              - 1
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & -1 \\ -1 & 0  \end{bmatrix}`
              - :math:`-X`
            * - ``"Y"``
              - 0
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & 1 \\ -1 & 0  \end{bmatrix}`
              - :math:`iY`
            * - ``"-Y"``
              - 1
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & -1 \\ 1 & 0  \end{bmatrix}`
              - :math:`-iY`
            * - ``"Z"``
              - 0
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`
              - :math:`Z`
            * - ``"-Z"``
              - 1
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} -1 & 0 \\ 0 & 1  \end{bmatrix}`
              - :math:`-Z`

        Args:
            array (bool): return a Numpy array if True, otherwise
                          return a list (Default: False).
            mode (Literal["S", "D", "B"]): return both stabilizer and destabilizer if "B",
                return only stabilizer if "S" and return only destabilizer if "D".

        Returns:
            list or array: The rows of the StabilizerTable in label form.
        Raises:
            QiskitError: if stabilizer and destabilizer are both False.
        """
        if mode not in ("S", "B", "D"):
            raise QiskitError("mode must be B, S, or D.")
        size = 2 * self.num_qubits if mode == "B" else self.num_qubits
        offset = self.num_qubits if mode == "S" else 0
        ret = np.zeros(size, dtype=f"<U{1 + self.num_qubits}")
        for i in range(size):
            z = self.tableau[i + offset, self.num_qubits : 2 * self.num_qubits]
            x = self.tableau[i + offset, 0 : self.num_qubits]
            phase = int(self.tableau[i + offset, -1]) * 2
            label = BasePauli._to_label(z, x, phase, group_phase=True)
            if label[0] != "-":
                label = "+" + label
            ret[i] = label
        if array:
            return ret
        return ret.tolist()

    # ---------------------------------------------------------------------
    # Internal helper functions
    # ---------------------------------------------------------------------

    def _hash(self):
        """Produce a hashable value that is unique for each different Clifford.  This should only be
        used internally when the classes being hashed are under our control, because classes of this
        type are mutable."""
        return np.packbits(self.tableau).tobytes()

    @staticmethod
    def _is_symplectic(mat):
        """Return True if input is symplectic matrix."""
        # Condition is
        # table.T * [[0, 1], [1, 0]] * table = [[0, 1], [1, 0]]
        # where we are block matrix multiplying using symplectic product

        dim = len(mat) // 2
        if mat.shape != (2 * dim, 2 * dim):
            return False

        one = np.eye(dim, dtype=int)
        zero = np.zeros((dim, dim), dtype=int)
        seye = np.block([[zero, one], [one, zero]])
        arr = mat.astype(int)
        return np.array_equal(np.mod(arr.T.dot(seye).dot(arr), 2), seye)

    @staticmethod
    def _conjugate_transpose(clifford, method):
        """Return the adjoint, conjugate, or transpose of the Clifford.

        Args:
            clifford (Clifford): a clifford object.
            method (str): what function to apply 'A', 'C', or 'T'.

        Returns:
            Clifford: the modified clifford.
        """
        ret = clifford.copy()
        if method in ["A", "T"]:
            # Apply inverse
            # Update table
            tmp = ret.destab_x.copy()
            ret.destab_x = ret.stab_z.T
            ret.destab_z = ret.destab_z.T
            ret.stab_x = ret.stab_x.T
            ret.stab_z = tmp.T
            # Update phase
            ret.phase ^= clifford.dot(ret).phase
        if method in ["C", "T"]:
            # Apply conjugate
            ret.phase ^= np.mod(_count_y(ret.x, ret.z), 2).astype(bool)
        return ret

    def _pad_with_identity(self, clifford, qargs):
        """Pad Clifford with identities on other subsystems."""
        if qargs is None:
            return clifford

        padded = Clifford(np.eye(2 * self.num_qubits, dtype=bool), validate=False, copy=False)
        inds = list(qargs) + [self.num_qubits + i for i in qargs]

        # Pad Pauli array
        for i, pos in enumerate(qargs):
            padded.tableau[inds, pos] = clifford.tableau[:, i]
            padded.tableau[inds, self.num_qubits + pos] = clifford.tableau[
                :, clifford.num_qubits + i
            ]

        # Pad phase
        padded.phase[inds] = clifford.phase

        return padded

    @staticmethod
    def _stack_table_phase(table, phase):
        return np.hstack((table, phase.reshape(len(phase), 1)))

    @staticmethod
    def _from_label(label):
        phase = False
        if label[0] in ("-", "+"):
            phase = label[0] == "-"
            label = label[1:]
        num_qubits = len(label)
        symp = np.zeros(2 * num_qubits + 1, dtype=bool)
        xs = symp[0:num_qubits]
        zs = symp[num_qubits : 2 * num_qubits]
        for i, char in enumerate(label):
            if char not in ["I", "X", "Y", "Z"]:
                raise QiskitError(
                    f"Pauli string contains invalid character: {char} not in ['I', 'X', 'Y', 'Z']."
                )
            if char in ("X", "Y"):
                xs[num_qubits - 1 - i] = True
            if char in ("Z", "Y"):
                zs[num_qubits - 1 - i] = True
        symp[-1] = phase
        return symp

    @staticmethod
    def _pauli_matrix_to_row(mat, num_qubits):
        """Generate a binary vector (a row of tableau representation) from a Pauli matrix.
        Return None if the non-Pauli matrix is supplied."""
        # pylint: disable=too-many-return-statements
        decimals = 6

        def find_one_index(x):
            indices = np.where(np.round(np.abs(x), decimals=decimals) == 1)
            return indices[0][0] if len(indices[0]) == 1 else None

        def bitvector(n, num_bits):
            return np.array([int(digit) for digit in format(n, f"0{num_bits}b")], dtype=bool)[::-1]

        # compute x-bits
        xint = find_one_index(mat[0, :])
        if xint is None:
            return None
        xbits = bitvector(xint, num_qubits)

        # extract non-zero elements from matrix (each must be 1, -1, 1j or -1j for Pauli matrix)
        entries = np.empty(len(mat), dtype=complex)
        for i, row in enumerate(mat):
            index = find_one_index(row)
            if index is None:
                return None
            expected = xint ^ i
            if index != expected:
                return None
            entries[i] = np.round(mat[i, index], decimals=decimals)
            if entries[i] not in {1, -1, 1j, -1j}:
                return None

        # compute z-bits
        zbits = np.empty(num_qubits, dtype=bool)
        for k in range(num_qubits):
            sign = np.round(entries[2**k] / entries[0])
            if sign == 1:
                zbits[k] = False
            elif sign == -1:
                zbits[k] = True
            else:
                return None

        # compute phase
        phase = None
        num_y = sum(xbits & zbits)
        positive_phase = (-1j) ** num_y
        if entries[0] == positive_phase:
            phase = False
        elif entries[0] == -1 * positive_phase:
            phase = True
        if phase is None:
            return None

        # validate all non-zero elements
        coef = ((-1) ** phase) * positive_phase
        ivec, zvec = np.ones(2), np.array([1, -1])
        expected = coef * functools.reduce(np.kron, [zvec if z else ivec for z in zbits[::-1]])
        if not np.allclose(entries, expected):
            return None

        return np.hstack([xbits, zbits, phase])

    @staticmethod
    def _unitary_matrix_to_tableau(matrix):
        # pylint: disable=invalid-name
        num_qubits = int(np.log2(len(matrix)))

        stab = np.empty((num_qubits, 2 * num_qubits + 1), dtype=bool)
        for i in range(num_qubits):
            label = "I" * (num_qubits - i - 1) + "X" + "I" * i
            Xi = Operator.from_label(label).to_matrix()
            target = matrix @ Xi @ np.conj(matrix).T
            row = Clifford._pauli_matrix_to_row(target, num_qubits)
            if row is None:
                return None
            stab[i] = row

        destab = np.empty((num_qubits, 2 * num_qubits + 1), dtype=bool)
        for i in range(num_qubits):
            label = "I" * (num_qubits - i - 1) + "Z" + "I" * i
            Zi = Operator.from_label(label).to_matrix()
            target = matrix @ Zi @ np.conj(matrix).T
            row = Clifford._pauli_matrix_to_row(target, num_qubits)
            if row is None:
                return None
            destab[i] = row

        tableau = np.vstack([stab, destab])
        return tableau


# Update docstrings for API docs
generate_apidocs(Clifford)
