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
# pylint: disable=invalid-name, abstract-method

import copy
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.pauli_tools import (
    coeff_phase_from_complex)


class BasePauli(BaseOperator):
    r"""Symplectic representation of a list of N-qubit Paulis.

    Base class for Pauli and PauliList.
    """

    def __init__(self, z, x, phase):
        """Initialize the BasePauli.

        This is an array of M N-qubit Paulis defined as
        P = (-i)^phase Z^z X^x.

        Args:
            z (array): input z matrix.
            x (array): input x matrix.
            phase (array): input phase vector.
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
        """Return the tensor product Pauli self ⊗ other.

        Args:
            other (BasePauli): another Pauli.

        Returns:
            BasePauli: the tensor product Pauli.
        """
        z = np.hstack([self._stack(other._z, self._num_paulis), self._z])
        x = np.hstack([self._stack(other._x, self._num_paulis), self._x])
        phase = self._phase + other._phase
        return BasePauli(z, x, phase)

    def expand(self, other):
        """Return the tensor product Pauli other ⊗ self.

        Args:
            other (BasePauli): another Pauli.

        Returns:
            BasePauli: the tensor product Pauli.
        """
        z = np.hstack([self._z, self._stack(other._z, self._num_paulis)])
        x = np.hstack([self._x, self._stack(other._x, self._num_paulis)])
        phase = self._phase + other._phase
        return BasePauli(z, x, phase)

    # pylint: disable=arguments-differ
    def compose(self, other, qargs=None, front=True, inplace=False):
        """Return the composition of Paulis self∘other.

        Note that this discards phases.

        Args:
            other (BasePauli): another Pauli.
            qargs (None or list): qubits to apply dot product on (default: None).
            front (bool): If True use `dot` composition method (default: False).
            inplace (bool): If True update in-place (default: False).

        Returns:
            BasePauli: the output Pauli.

        Raises:
            QiskitError: if number of qubits of other does not match qargs.
        """
        # Validation
        if qargs is None and other.num_qubits != self.num_qubits:
            raise QiskitError(
                "other {} must be on the same number of qubits.".format(
                    type(self).__name__))

        if qargs and other.num_qubits != len(qargs):
            raise QiskitError(
                "Number of qubits of the other {} does not match qargs.".format(
                    type(self).__name__))

        if len(other) not in [1, len(self)]:
            raise QiskitError("Incompatible BasePaulis. Second list must "
                              "either have 1 or the same number of Paulis.")

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
        ret._phase = phase
        return ret

    # pylint: disable=arguments-differ
    def dot(self, other, qargs=None, inplace=False):
        """Return the dot product of Paulis self∘other.

        Note that this discards phases.

        Args:
            other (BasePauli): another Pauli.
            qargs (None or list): qubits to apply dot product on (default: None).
            inplace (bool): If True update in-place (default: False).

        Returns:
            BasePauli: the output Pauli.

        Raises:
            QiskitError: if number of qubits of other does not match qargs.
        """
        return self.compose(other, qargs=qargs, front=True, inplace=inplace)

    def _multiply(self, other):
        """Multiply each Pauli in the table by a phase.

        Args:
            other (complex): a complex number in [1, -1j, -1, 1j]

        Returns:
            BasePauli: the Pauli table other * self.

        Raises:
            QiskitError: if the phase is not in the set [1, -1j, -1, 1j].
        """
        if isinstance(other, (np.ndarray, list, tuple)):
            phase = np.array([coeff_phase_from_complex(phase) for phase in other])
        else:
            phase = coeff_phase_from_complex(other)
        return BasePauli(self._z, self._x, self._phase + phase)

    def conjugate(self):
        """Return the conjugate of each Pauli in the list."""
        complex_phase = np.mod(self._phase, 2)
        if np.all(complex_phase == 0):
            return self
        return BasePauli(self._z, self._x, self._phase + 2 * complex_phase)

    def transpose(self):
        """Return the transpose of each Pauli in the list."""
        # Transpose sets Y -> -Y. This has effect on changing the phase
        parity_y = self._count_y() % 2
        if np.all(parity_y == 0):
            return self
        return BasePauli(self._z, self._x, self._phase + 2 * parity_y)

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
                "qargs ({} != {}).".format(other.num_qubits, len(qargs)))
        if qargs is None and self.num_qubits != other.num_qubits:
            raise QiskitError(
                "Number of qubits of other Pauli does not match the current "
                "Pauli ({} != {}).".format(other.num_qubits, self.num_qubits))
        if qargs is not None:
            inds = list(qargs)
            x1, z1 = self._x[:, inds], self._z[:, inds]
        else:
            x1, z1 = self._x, self._z
        a_dot_b = np.mod(np.sum(np.logical_and(x1, other._z), axis=1), 2)
        b_dot_a = np.mod(np.sum(np.logical_and(z1, other._x), axis=1), 2)
        return a_dot_b == b_dot_a

    def evolve(self, other, qargs=None):
        r"""Evolve Pauli by a Clifford.

        This returns the Pauli :math:`P^\prime = C.P.C^\dagger`.

        Args:
            other (BasePauli or QuantumCircuit): The Clifford circuit to evolve by.
            qargs (list): a list of qubits to apply the Clifford to.

        Returns:
            BasePauli: the Pauli :math:`C.P.C^\dagger`.

        Raises:
            QiskitError: if the Clifford number of qubits and qargs don't match.
        """
        # Check dimension
        if qargs is not None and len(qargs) != other.num_qubits:
            raise QiskitError(
                "Incorrect number of qubits for Clifford circuit ({} != {}).".format(
                    other.num_qubits, len(qargs)))
        if qargs is None and self.num_qubits != other.num_qubits:
            raise QiskitError(
                "Incorrect number of qubits for Clifford circuit ({} != {}).".format(
                    other.num_qubits, self.num_qubits))

        # Evolve via Pauli
        if isinstance(other, BasePauli):
            ret = self.compose(other, qargs=qargs)
            ret = ret.dot(other.adjoint(), qargs=qargs)
            return ret

        # Otherwise evolve by circuit evolution
        ret = self.copy()
        _evolve_circuit(ret, other, qargs=qargs)
        return ret

    # ---------------------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------------------

    def __imul__(self, other):
        return self.dot(other, inplace=True)

    def __neg__(self):
        ret = copy.copy(self)
        ret._phase = np.mod(self._phase + 2, 4)
        return ret

    def _count_y(self):
        """Count the number of I Pauli's"""
        return np.sum(np.logical_and(self._x, self._z), axis=1)

    @staticmethod
    def _stack(array, size):
        """Stack array."""
        if size == 1:
            return array
        if array.ndim == 1:
            return np.concatenate(size * [array])
        return np.vstack(size * [array])

    @staticmethod
    def _block_stack(array1, array2):
        """Stack two arrays along their first axis."""
        sz1 = len(array1)
        sz2 = len(array2)
        out_shape1 = (sz1 * sz2, ) + array1.shape[1:]
        out_shape2 = (sz1 * sz2, ) + array2.shape[1:]
        if sz2 > 1:
            # Stack blocks for output table
            ret1 = np.reshape(np.stack(sz2 * [array1], axis=1),
                              out_shape1)
        else:
            ret1 = array1
        if sz1 > 1:
            # Stack blocks for output table
            ret2 = np.reshape(np.vstack(sz1 * [array2]), out_shape2)
        else:
            ret2 = array2
        return ret1, ret2


# ---------------------------------------------------------------------
# Pauli Evolution by Clifford circuit
# ---------------------------------------------------------------------

def _evolve_circuit(base_pauli, circuit, qargs=None):
    """Update Pauli inplace by applying a Clifford circuit.

    Args:
        base_pauli (BasePauli): the Pauli or PauliList to update.
        circuit (QuantumCircuit or Instruction): the gate or composite gate to apply.
        qargs (list or None): The qubits to apply gate to.

    Returns:
        BasePauli: the updated Pauli.

    Raises:
        QiskitError: if input gate cannot be decomposed into Clifford gates.
    """
    if isinstance(circuit, Barrier):
        return base_pauli

    if qargs is None:
        qargs = list(range(base_pauli.num_qubits))

    if isinstance(circuit, QuantumCircuit):
        phase = float(circuit.global_phase)
        if phase:
            base_pauli._phase += coeff_phase_from_complex(np.exp(1j * phase))
        gate = circuit.to_instruction()
    else:
        gate = circuit

    # Basis Clifford Gates
    basis_1q = {
        'i': _evolve_i,
        'id': _evolve_i,
        'iden': _evolve_i,
        'x': _evolve_x,
        'y': _evolve_y,
        'z': _evolve_z,
        'h': _evolve_h,
        's': _evolve_s,
        'sdg': _evolve_sdg,
        'sinv': _evolve_sdg
    }
    basis_2q = {
        'cx': _evolve_cx,
        'cz': _evolve_cz,
        'cy': _evolve_cy,
        'swap': _evolve_swap
    }

    # Non-clifford gates
    non_clifford = ['t', 'tdg', 'ccx', 'ccz']

    if isinstance(gate, str):
        # Check if gate is a valid Clifford basis gate string
        if gate not in basis_1q and gate not in basis_2q:
            raise QiskitError(
                "Invalid Clifford gate name string {}".format(gate))
        name = gate
    else:
        # Assume gate is an Instruction
        name = gate.name

    # Apply gate if it is a Clifford basis gate
    if name in non_clifford:
        raise QiskitError(
            "Cannot update Pauli with non-Clifford gate {}".format(name))
    if name in basis_1q:
        if len(qargs) != 1:
            raise QiskitError("Invalid qubits for 1-qubit gate.")
        return basis_1q[name](base_pauli, qargs[0])
    if name in basis_2q:
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate.")
        return basis_2q[name](base_pauli, qargs[0], qargs[1])

    # If not a Clifford basis gate we try to unroll the gate and
    # raise an exception if unrolling reaches a non-Clifford gate.
    # TODO: We could also check u3 params to see if they
    # are a single qubit Clifford gate rather than raise an exception.
    if gate.definition is None:
        raise QiskitError('Cannot apply Instruction: {}'.format(gate.name))
    if not isinstance(gate.definition, QuantumCircuit):
        raise QiskitError(
            '{} instruction definition is {}; expected QuantumCircuit'.format(
                gate.name, type(gate.definition)))
    for instr, qregs, cregs in gate.definition:
        if cregs:
            raise QiskitError(
                'Cannot apply Instruction with classical registers: {}'.format(
                    instr.name))
        # Get the integer position of the flat register
        new_qubits = [qargs[tup.index] for tup in qregs]
        _evolve_circuit(base_pauli, instr, new_qubits)
    return base_pauli


def _evolve_h(base_pauli, qubit):
    """Evolve by HGate"""
    x = base_pauli._x[:, qubit].copy()
    z = base_pauli._z[:, qubit].copy()
    base_pauli._x[:, qubit] = z
    base_pauli._z[:, qubit] = x
    base_pauli._phase = np.mod(
        base_pauli._phase + 2 * np.logical_and(x, z).T, 4)
    return base_pauli


def _evolve_s(base_pauli, qubit):
    """Evolve by SGate"""
    x = base_pauli._x[:, qubit]
    base_pauli._z[:, qubit] ^= x
    base_pauli._phase += x.T
    return base_pauli


def _evolve_sdg(base_pauli, qubit):
    """Evolve by SdgGate"""
    x = base_pauli._x[:, qubit]
    base_pauli._z[:, qubit] ^= x
    base_pauli._phase -= x.T
    return base_pauli


# pylint: disable=unused-argument
def _evolve_i(base_pauli, qubit):
    """Evolve by IGate"""
    return base_pauli


def _evolve_x(base_pauli, qubit):
    """Evolve by XGate"""
    base_pauli._phase += 2 * base_pauli._z[:, qubit].T
    return base_pauli


def _evolve_y(base_pauli, qubit):
    """Evolve by YGate"""
    base_pauli._phase += 2 * base_pauli._x[:, qubit].T + 2 * base_pauli._z[:, qubit].T
    return base_pauli


def _evolve_z(base_pauli, qubit):
    """Evolve by ZGate"""
    base_pauli._phase += 2 * base_pauli._x[:, qubit].T
    return base_pauli


def _evolve_cx(base_pauli, qctrl, qtrgt):
    """Evolve by CXGate"""
    base_pauli._x[:, qtrgt] ^= base_pauli._x[:, qctrl]
    base_pauli._z[:, qctrl] ^= base_pauli._z[:, qtrgt]
    return base_pauli


def _evolve_cz(base_pauli, q1, q2):
    """Evolve by CZGate"""
    x1 = base_pauli._x[:, q1]
    x2 = base_pauli._x[:, q2]
    base_pauli._z[:, q1] ^= x1
    base_pauli._z[:, q2] ^= x2
    base_pauli._phase += 2 * np.logical_and(x1, x2).T
    return base_pauli


def _evolve_cy(base_pauli, qctrl, qtrgt):
    """Evolve by CYGate"""
    x1 = base_pauli._x[:, qctrl]
    x2 = base_pauli._x[:, qtrgt]
    z2 = base_pauli._z[:, qtrgt]
    base_pauli._x[:, qtrgt] ^= x1
    base_pauli._z[:, qtrgt] ^= x1
    base_pauli._z[:, qctrl] ^= np.logical_xor(x2, z2)
    base_pauli._phase += x1 + 2 * np.logical_and(x1, x2).T
    return base_pauli


def _evolve_swap(base_pauli, q1, q2):
    """Evolve by SwapGate"""
    x1 = base_pauli._x[:, q1]
    z1 = base_pauli._z[:, q1]
    base_pauli._x[:, q1] = base_pauli._x[:, q2]
    base_pauli._z[:, q1] = base_pauli._z[:, q2]
    base_pauli._x[:, q2] = x1
    base_pauli._z[:, q2] = z1
    return base_pauli
