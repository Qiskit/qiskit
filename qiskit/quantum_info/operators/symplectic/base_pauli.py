# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Optimized list of Pauli operators
"""
# pylint: disable=invalid-name

import copy

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, MultiplyMixin


class BasePauli(BaseOperator, AdjointMixin, MultiplyMixin):
    r"""Symplectic representation of a list of N-qubit Paulis.

    Base class for Pauli and PauliList.
    """

    def __init__(self, z, x, phase):
        """Initialize the BasePauli.

        This is an array of M N-qubit Paulis defined as
        P = (-i)^phase Z^z X^x.

        Args:
            z (np.ndarray): input z matrix.
            x (np.ndarray): input x matrix.
            phase (np.ndarray): input phase vector.
        """
        self._z = z
        self._x = x
        self._phase = phase
        self._num_paulis, num_qubits = self._z.shape
        super().__init__(num_qubits=num_qubits)

    def copy(self):
        """Make a deep copy of current operator."""
        # Deepcopy has terrible performance on objects with Numpy arrays
        # attributes so we make a shallow copy and then manually copy the
        # Numpy arrays to efficiently mimic a deepcopy
        ret = copy.copy(self)
        ret._z = self._z.copy()
        ret._x = self._x.copy()
        ret._phase = self._phase.copy()
        return ret

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def tensor(self, other):
        return self._tensor(self, other)

    def expand(self, other):
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        x1 = cls._stack(a._x, b._num_paulis, False)
        x2 = cls._stack(b._x, a._num_paulis)
        z1 = cls._stack(a._z, b._num_paulis, False)
        z2 = cls._stack(b._z, a._num_paulis)
        phase1 = (
            np.vstack(b._num_paulis * [a._phase])
            .transpose(1, 0)
            .reshape(a._num_paulis * b._num_paulis)
        )
        phase2 = cls._stack(b._phase, a._num_paulis)
        z = np.hstack([z2, z1])
        x = np.hstack([x2, x1])
        phase = np.mod(phase1 + phase2, 4)
        return BasePauli(z, x, phase)

    # pylint: disable=arguments-differ
    def compose(self, other, qargs=None, front=False, inplace=False):
        """Return the composition of Paulis.

        Args:
            a ({cls}): an operator object.
            b ({cls}): an operator object.
            qargs (list or None): Optional, qubits to apply dot product
                                  on (default: None).
            inplace (bool): If True update in-place (default: False).

        Returns:
            {cls}: The operator a.compose(b)

        Raises:
            QiskitError: if number of qubits of other does not match qargs.
        """.format(
            cls=type(self).__name__
        )
        # Validation
        if qargs is None and other.num_qubits != self.num_qubits:
            raise QiskitError(f"other {type(self).__name__} must be on the same number of qubits.")

        if qargs and other.num_qubits != len(qargs):
            raise QiskitError(
                f"Number of qubits of the other {type(self).__name__} does not match qargs."
            )

        if other._num_paulis not in [1, self._num_paulis]:
            raise QiskitError(
                "Incompatible BasePaulis. Second list must "
                "either have 1 or the same number of Paulis."
            )

        # Compute phase shift
        if qargs is not None:
            x1, z1 = self._x[:, qargs], self._z[:, qargs]
        else:
            x1, z1 = self._x, self._z
        x2, z2 = other._x, other._z

        # Get phase shift
        phase = self._phase + other._phase
        if front:
            phase += 2 * np.sum(np.logical_and(x1, z2), axis=1)
        else:
            phase += 2 * np.sum(np.logical_and(z1, x2), axis=1)

        # Update Pauli
        x = np.logical_xor(x1, x2)
        z = np.logical_xor(z1, z2)

        if qargs is None:
            if not inplace:
                return BasePauli(z, x, phase)
            # Inplace update
            self._x = x
            self._z = z
            self._phase = phase
            return self

        # Qargs update
        ret = self if inplace else self.copy()
        ret._x[:, qargs] = x
        ret._z[:, qargs] = z
        ret._phase = np.mod(phase, 4)
        return ret

    def _multiply(self, other):
        """Return the {cls} other * self.

        Args:
            other (complex): a complex number in ``[1, -1j, -1, 1j]``.

        Returns:
            {cls}: the {cls} other * self.

        Raises:
            QiskitError: if the phase is not in the set ``[1, -1j, -1, 1j]``.
        """.format(
            cls=type(self).__name__
        )
        if isinstance(other, (np.ndarray, list, tuple)):
            phase = np.array([self._phase_from_complex(phase) for phase in other])
        else:
            phase = self._phase_from_complex(other)
        return BasePauli(self._z, self._x, np.mod(self._phase + phase, 4))

    def conjugate(self):
        """Return the conjugate of each Pauli in the list."""
        complex_phase = np.mod(self._phase, 2)
        if np.all(complex_phase == 0):
            return self
        return BasePauli(self._z, self._x, np.mod(self._phase + 2 * complex_phase, 4))

    def transpose(self):
        """Return the transpose of each Pauli in the list."""
        # Transpose sets Y -> -Y. This has effect on changing the phase
        parity_y = self._count_y() % 2
        if np.all(parity_y == 0):
            return self
        return BasePauli(self._z, self._x, np.mod(self._phase + 2 * parity_y, 4))

    def commutes(self, other, qargs=None):
        """Return True if Pauli that commutes with other.

        Args:
            other (BasePauli): another BasePauli operator.
            qargs (list): qubits to apply dot product on (default: None).

        Returns:
            np.array: Boolean array of True if Pauli's commute, False if
                      they anti-commute.

        Raises:
            QiskitError: if number of qubits of other does not match qargs.
        """
        if qargs is not None and len(qargs) != other.num_qubits:
            raise QiskitError(
                "Number of qubits of other Pauli does not match number of "
                "qargs ({} != {}).".format(other.num_qubits, len(qargs))
            )
        if qargs is None and self.num_qubits != other.num_qubits:
            raise QiskitError(
                "Number of qubits of other Pauli does not match the current "
                "Pauli ({} != {}).".format(other.num_qubits, self.num_qubits)
            )
        if qargs is not None:
            inds = list(qargs)
            x1, z1 = self._x[:, inds], self._z[:, inds]
        else:
            x1, z1 = self._x, self._z
        a_dot_b = np.mod(np.sum(np.logical_and(x1, other._z), axis=1), 2)
        b_dot_a = np.mod(np.sum(np.logical_and(z1, other._x), axis=1), 2)
        return a_dot_b == b_dot_a

    def evolve(self, other, qargs=None):
        r"""Heisenberg picture evolution of a Pauli by a Clifford.

        This returns the Pauli :math:`P^\prime = C^\dagger.P.C`.

        Args:
            other (BasePauli or QuantumCircuit): The Clifford circuit to evolve by.
            qargs (list): a list of qubits to apply the Clifford to.

        Returns:
            BasePauli: the Pauli :math:`C^\dagger.P.C`.

        Raises:
            QiskitError: if the Clifford number of qubits and qargs don't match.
        """
        # Check dimension
        if qargs is not None and len(qargs) != other.num_qubits:
            raise QiskitError(
                "Incorrect number of qubits for Clifford circuit ({} != {}).".format(
                    other.num_qubits, len(qargs)
                )
            )
        if qargs is None and self.num_qubits != other.num_qubits:
            raise QiskitError(
                "Incorrect number of qubits for Clifford circuit ({} != {}).".format(
                    other.num_qubits, self.num_qubits
                )
            )

        # Evolve via Pauli
        if isinstance(other, BasePauli):
            ret = self.compose(other.adjoint(), qargs=qargs)
            ret = ret.compose(other, front=True, qargs=qargs)
            return ret

        # pylint: disable=cyclic-import
        from qiskit.quantum_info.operators.symplectic.clifford import Clifford

        # Convert Clifford to quantum circuits
        if isinstance(other, Clifford):
            return self._evolve_clifford(other, qargs=qargs)

        # Otherwise evolve by the inverse circuit to compute C^dg.P.C
        return self.copy()._append_circuit(other.inverse(), qargs=qargs)

    def _evolve_clifford(self, other, qargs=None):
        """Heisenberg picture evolution of a Pauli by a Clifford."""
        if qargs is None:
            idx = slice(None)
            num_act = self.num_qubits
        else:
            idx = list(qargs)
            num_act = len(idx)

        # Set return to I on qargs
        ret = self.copy()
        ret._x[:, idx] = False
        ret._z[:, idx] = False

        # pylint: disable=cyclic-import
        from qiskit.quantum_info.operators.symplectic.pauli import Pauli
        from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList

        # Get action of Pauli's from Clifford
        adj = other.adjoint()
        pauli_list = []
        for z in self._z[:, idx]:
            pauli = Pauli("I" * num_act)
            for row in adj.stabilizer[z]:
                pauli.compose(Pauli((row.Z[0], row.X[0], 2 * row.phase[0])), inplace=True)
            pauli_list.append(pauli)
        ret.dot(PauliList(pauli_list), qargs=qargs, inplace=True)

        pauli_list = []
        for x in self._x[:, idx]:
            pauli = Pauli("I" * num_act)
            for row in adj.destabilizer[x]:
                pauli.compose(Pauli((row.Z[0], row.X[0], 2 * row.phase[0])), inplace=True)
            pauli_list.append(pauli)
        ret.dot(PauliList(pauli_list), qargs=qargs, inplace=True)
        return ret

    def _eq(self, other):
        """Entrywise comparison of Pauli equality."""
        return (
            self.num_qubits == other.num_qubits
            and np.all(np.mod(self._phase, 4) == np.mod(other._phase, 4))
            and np.all(self._z == other._z)
            and np.all(self._x == other._x)
        )

    # ---------------------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------------------

    def __imul__(self, other):
        return self.compose(other, front=True, inplace=True)

    def __neg__(self):
        ret = copy.copy(self)
        ret._phase = np.mod(self._phase + 2, 4)
        return ret

    def _count_y(self):
        """Count the number of I Pauli's"""
        return np.sum(np.logical_and(self._x, self._z), axis=1)

    @staticmethod
    def _stack(array, size, vertical=True):
        """Stack array."""
        if size == 1:
            return array
        if vertical:
            return np.vstack(size * [array]).reshape((size * len(array),) + array.shape[1:])
        return np.hstack(size * [array]).reshape((size * len(array),) + array.shape[1:])

    @staticmethod
    def _phase_from_complex(coeff):
        """Return the phase from a label"""
        if np.isclose(coeff, 1):
            return 0
        if np.isclose(coeff, -1j):
            return 1
        if np.isclose(coeff, -1):
            return 2
        if np.isclose(coeff, 1j):
            return 3
        raise QiskitError("Pauli can only be multiplied by 1, -1j, -1, 1j.")

    @staticmethod
    def _from_array(z, x, phase=0):
        """Convert array data to BasePauli data."""
        if isinstance(z, np.ndarray) and z.dtype == bool:
            base_z = z
        else:
            base_z = np.asarray(z, dtype=bool)
        if base_z.ndim == 1:
            base_z = base_z.reshape((1, base_z.size))
        elif base_z.ndim != 2:
            raise QiskitError("Invalid Pauli z vector shape.")

        if isinstance(x, np.ndarray) and x.dtype == bool:
            base_x = x
        else:
            base_x = np.asarray(x, dtype=bool)
        if base_x.ndim == 1:
            base_x = base_x.reshape((1, base_x.size))
        elif base_x.ndim != 2:
            raise QiskitError("Invalid Pauli x vector shape.")

        if base_z.shape != base_x.shape:
            raise QiskitError("z and x vectors are different size.")

        # Convert group phase convention to internal ZX-phase conversion.
        base_phase = np.mod(np.sum(np.logical_and(base_x, base_z), axis=1, dtype=int) + phase, 4)
        return base_z, base_x, base_phase

    @staticmethod
    def _to_matrix(z, x, phase=0, group_phase=False, sparse=False):
        """Return the matrix matrix from symplectic representation.

        The Pauli is defined as :math:`P = (-i)^{phase + z.x} * Z^z.x^x`
        where ``array = [x, z]``.

        Args:
            z (array): The symplectic representation z vector.
            x (array): The symplectic representation x vector.
            phase (int): Pauli phase.
            group_phase (bool): Optional. If True use group-phase convention
                                instead of BasePauli ZX-phase convention.
                                (default: False).
            sparse (bool): Optional. Of True return a sparse CSR matrix,
                           otherwise return a dense Numpy array
                           (default: False).

        Returns:
            array: if sparse=False.
            csr_matrix: if sparse=True.
        """
        num_qubits = z.size

        # Convert to zx_phase
        if group_phase:
            phase += np.sum(x & z)
            phase %= 4

        dim = 2 ** num_qubits
        twos_array = 1 << np.arange(num_qubits)
        x_indices = np.asarray(x).dot(twos_array)
        z_indices = np.asarray(z).dot(twos_array)

        indptr = np.arange(dim + 1, dtype=np.uint)
        indices = indptr ^ x_indices
        if phase:
            coeff = (-1j) ** phase
        else:
            coeff = 1
        data = np.array([coeff * (-1) ** (bin(i).count("1") % 2) for i in z_indices & indptr])
        if sparse:
            # Return sparse matrix
            from scipy.sparse import csr_matrix

            return csr_matrix((data, indices, indptr), shape=(dim, dim), dtype=complex)

        # Build dense matrix using csr format
        mat = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            mat[i][indices[indptr[i] : indptr[i + 1]]] = data[indptr[i] : indptr[i + 1]]
        return mat

    @staticmethod
    def _to_label(z, x, phase, group_phase=False, full_group=True, return_phase=False):
        """Return the label string for a Pauli.

        Args:
            z (array): The symplectic representation z vector.
            x (array): The symplectic representation x vector.
            phase (int): Pauli phase.
            group_phase (bool): Optional. If True use group-phase convention
                                instead of BasePauli ZX-phase convention.
                                (default: False).
            full_group (bool): If True return the Pauli label from the full Pauli group
                including complex coefficient from [1, -1, 1j, -1j]. If
                False return the unsigned Pauli label with coefficient 1
                (default: True).
            return_phase (bool): If True return the adjusted phase for the coefficient
                of the returned Pauli label. This can be used even if
                ``full_group=False``.

        Returns:
            str: the Pauli label from the full Pauli group (if ``full_group=True``) or
                from the unsigned Pauli group (if ``full_group=False``).
            Tuple[str, int]: if ``return_phase=True`` returns a tuple of the Pauli
                            label (from either the full or unsigned Pauli group) and
                            the phase ``q`` for the coefficient :math:`(-i)^(q + x.z)`
                            for the label from the full Pauli group.
        """
        num_qubits = z.size
        coeff_labels = {0: "", 1: "-i", 2: "-", 3: "i"}
        label = ""
        for i in range(num_qubits):
            if not z[num_qubits - 1 - i]:
                if not x[num_qubits - 1 - i]:
                    label += "I"
                else:
                    label += "X"
            elif not x[num_qubits - 1 - i]:
                label += "Z"
            else:
                label += "Y"
                if not group_phase:
                    phase -= 1
        phase %= 4
        if phase and full_group:
            label = coeff_labels[phase] + label
        if return_phase:
            return label, phase
        return label

    def _append_circuit(self, circuit, qargs=None):
        """Update BasePauli inplace by applying a Clifford circuit.

        Args:
            circuit (QuantumCircuit or Instruction): the gate or composite gate to apply.
            qargs (list or None): The qubits to apply gate to.

        Returns:
            BasePauli: the updated Pauli.

        Raises:
            QiskitError: if input gate cannot be decomposed into Clifford gates.
        """
        if isinstance(circuit, Barrier):
            return self

        if qargs is None:
            qargs = list(range(self.num_qubits))

        if isinstance(circuit, QuantumCircuit):
            gate = circuit.to_instruction()
        else:
            gate = circuit

        # Basis Clifford Gates
        basis_1q = {
            "i": _evolve_i,
            "id": _evolve_i,
            "iden": _evolve_i,
            "x": _evolve_x,
            "y": _evolve_y,
            "z": _evolve_z,
            "h": _evolve_h,
            "s": _evolve_s,
            "sdg": _evolve_sdg,
            "sinv": _evolve_sdg,
        }
        basis_2q = {"cx": _evolve_cx, "cz": _evolve_cz, "cy": _evolve_cy, "swap": _evolve_swap}

        # Non-Clifford gates
        non_clifford = ["t", "tdg", "ccx", "ccz"]

        if isinstance(gate, str):
            # Check if gate is a valid Clifford basis gate string
            if gate not in basis_1q and gate not in basis_2q:
                raise QiskitError(f"Invalid Clifford gate name string {gate}")
            name = gate
        else:
            # Assume gate is an Instruction
            name = gate.name

        # Apply gate if it is a Clifford basis gate
        if name in non_clifford:
            raise QiskitError(f"Cannot update Pauli with non-Clifford gate {name}")
        if name in basis_1q:
            if len(qargs) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate.")
            return basis_1q[name](self, qargs[0])
        if name in basis_2q:
            if len(qargs) != 2:
                raise QiskitError("Invalid qubits for 2-qubit gate.")
            return basis_2q[name](self, qargs[0], qargs[1])

        # If not a Clifford basis gate we try to unroll the gate and
        # raise an exception if unrolling reaches a non-Clifford gate.
        if gate.definition is None:
            raise QiskitError(f"Cannot apply Instruction: {gate.name}")
        if not isinstance(gate.definition, QuantumCircuit):
            raise QiskitError(
                "{} instruction definition is {}; expected QuantumCircuit".format(
                    gate.name, type(gate.definition)
                )
            )

        flat_instr = gate.definition
        bit_indices = {
            bit: index
            for bits in [flat_instr.qubits, flat_instr.clbits]
            for index, bit in enumerate(bits)
        }

        for instr, qregs, cregs in flat_instr:
            if cregs:
                raise QiskitError(
                    f"Cannot apply Instruction with classical registers: {instr.name}"
                )
            # Get the integer position of the flat register
            new_qubits = [qargs[bit_indices[tup]] for tup in qregs]
            self._append_circuit(instr, new_qubits)

        # Since the individual gate evolution functions don't take mod
        # of phase we update it at the end
        self._phase %= 4
        return self


# ---------------------------------------------------------------------
# Evolution by Clifford gates
# ---------------------------------------------------------------------


def _evolve_h(base_pauli, qubit):
    """Update P -> H.P.H"""
    x = base_pauli._x[:, qubit].copy()
    z = base_pauli._z[:, qubit].copy()
    base_pauli._x[:, qubit] = z
    base_pauli._z[:, qubit] = x
    base_pauli._phase += 2 * np.logical_and(x, z).T
    return base_pauli


def _evolve_s(base_pauli, qubit):
    """Update P -> S.P.Sdg"""
    x = base_pauli._x[:, qubit]
    base_pauli._z[:, qubit] ^= x
    base_pauli._phase += x.T
    return base_pauli


def _evolve_sdg(base_pauli, qubit):
    """Update P -> Sdg.P.S"""
    x = base_pauli._x[:, qubit]
    base_pauli._z[:, qubit] ^= x
    base_pauli._phase -= x.T
    return base_pauli


# pylint: disable=unused-argument
def _evolve_i(base_pauli, qubit):
    """Update P -> P"""
    return base_pauli


def _evolve_x(base_pauli, qubit):
    """Update P -> X.P.X"""
    base_pauli._phase += 2 * base_pauli._z[:, qubit].T
    return base_pauli


def _evolve_y(base_pauli, qubit):
    """Update P -> Y.P.Y"""
    base_pauli._phase += 2 * base_pauli._x[:, qubit].T + 2 * base_pauli._z[:, qubit].T
    return base_pauli


def _evolve_z(base_pauli, qubit):
    """Update P -> Z.P.Z"""
    base_pauli._phase += 2 * base_pauli._x[:, qubit].T
    return base_pauli


def _evolve_cx(base_pauli, qctrl, qtrgt):
    """Update P -> CX.P.CX"""
    base_pauli._x[:, qtrgt] ^= base_pauli._x[:, qctrl]
    base_pauli._z[:, qctrl] ^= base_pauli._z[:, qtrgt]
    return base_pauli


def _evolve_cz(base_pauli, q1, q2):
    """Update P -> CZ.P.CZ"""
    x1 = base_pauli._x[:, q1].copy()
    x2 = base_pauli._x[:, q2].copy()
    base_pauli._z[:, q1] ^= x2
    base_pauli._z[:, q2] ^= x1
    base_pauli._phase += 2 * np.logical_and(x1, x2).T
    return base_pauli


def _evolve_cy(base_pauli, qctrl, qtrgt):
    """Update P -> CY.P.CY"""
    x1 = base_pauli._x[:, qctrl].copy()
    x2 = base_pauli._x[:, qtrgt].copy()
    z2 = base_pauli._z[:, qtrgt].copy()
    base_pauli._x[:, qtrgt] ^= x1
    base_pauli._z[:, qtrgt] ^= x1
    base_pauli._z[:, qctrl] ^= np.logical_xor(x2, z2)
    base_pauli._phase += x1 + 2 * np.logical_and(x1, x2).T
    return base_pauli


def _evolve_swap(base_pauli, q1, q2):
    """Update P -> SWAP.P.SWAP"""
    x1 = base_pauli._x[:, q1].copy()
    z1 = base_pauli._z[:, q1].copy()
    base_pauli._x[:, q1] = base_pauli._x[:, q2]
    base_pauli._z[:, q1] = base_pauli._z[:, q2]
    base_pauli._x[:, q2] = x1
    base_pauli._z[:, q2] = z1
    return base_pauli
