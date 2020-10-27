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
Helper methods for Pauli class.
"""
# pylint: disable=invalid-name

import re
from typing import Union, Tuple
import numpy as np
from scipy.sparse import csr_matrix

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier


# ---------------------------------------------------------------------
# Conversion helper functions
# ---------------------------------------------------------------------

def coeff_phase_from_complex(coeff: complex) -> int:
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


def split_pauli_label(label: str) -> Tuple[str, str]:
    """Split Pauli label into unsigned group label and coefficient label"""
    span = re.search(r'[IXYZ]+', label).span()
    pauli = label[span[0]:]
    coeff = label[:span[0]]
    if span[1] != len(label):
        raise QiskitError('Pauli string is not valid.')
    return pauli, coeff


def coeff_phase_from_label(label: str) -> Union[int, None]:
    """Return the phase from a label"""
    # Returns None if label is invalid
    label = label.replace('+', '', 1).replace('1', '', 1).replace('j', 'i', 1)
    phases = {'': 0, '-i': 1, '-': 2, 'i': 3}
    return phases.get(label)


def phase_to_coeff_label(phase: int) -> str:
    """Return the label from phase"""
    labels = {0: '', 1: '-i', 2: '-', 3: 'i'}
    return labels[phase % 4]


def pauli_from_label(label: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return the symplectic representation of Pauli string.

    Args:
        label: the Pauli string label.

    Returns:
        (z, x, q): the z vector, x vector, and phase integer q for the
                   symplectic representation P = (-i)^{q + z.x} Z^z.x^x.

    Raises:
        QiskitError: if Pauli string is not valid.
    """
    # Split string into coefficient and Pauli
    span = re.search(r'[IXYZ]+', label).span()
    pauli, coeff = split_pauli_label(label)
    coeff = label[:span[0]]

    # Convert coefficient to phase
    phase = 0 if not coeff else coeff_phase_from_label(coeff)
    if phase is None:
        raise QiskitError('Pauli string is not valid.')

    # Convert to Symplectic representation
    num_qubits = len(pauli)
    z = np.zeros(num_qubits, dtype=np.bool)
    x = np.zeros(num_qubits, dtype=np.bool)
    for i, char in enumerate(pauli):
        if char == 'X':
            x[num_qubits - 1 - i] = True
        elif char == 'Z':
            z[num_qubits - 1 - i] = True
        elif char == 'Y':
            x[num_qubits - 1 - i] = True
            z[num_qubits - 1 - i] = True
    return z, x, phase % 4


def pauli_to_label(z: np.ndarray,
                   x: np.ndarray,
                   phase: int = 0,
                   full_group: bool = True,
                   return_phase: bool = False) -> Union[str, Tuple[str, int]]:
    """Return the label string for a Pauli.

    Args:
        z: The symplectic representation z vector.
        x: The symplectic representation x vector.
        phase: the phase coefficient :math:`(-i)^q` of the Pauli.
        full_group: If True return the Pauli label from the full Pauli group
                    including complex coefficient from [1, -1, 1j, -1j]. If
                    False return the unsigned Pauli label with coefficient 1
                    (default: True).
        return_phase: If True return the adjusted phase for the coefficient
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
    label = ''
    for i in range(num_qubits):
        if not z[num_qubits - 1 - i]:
            if not x[num_qubits - 1 - i]:
                label += 'I'
            else:
                label += 'X'
        elif not x[num_qubits - 1 - i]:
            label += 'Z'
        else:
            label += 'Y'
    if phase and full_group:
        label = phase_to_coeff_label(phase) + label
    if return_phase:
        return label, phase
    return label


def pauli_to_matrix(z: np.ndarray,
                    x: np.ndarray,
                    phase: int = 0,
                    sparse: bool =False) -> Union[np.ndarray, csr_matrix]:
    """Return the matrix matrix from symplectic representation.

    The Pauli is defined as :math:`P = (-i)^{phase + z.x} * Z^z.x^x`
    where ``array = [x, z]``.

    Args:
        z: The symplectic representation z vector.
        x: The symplectic representation x vector.
        phase: Pauli phase.
        sparse: if True return a sparse CSR matrix, otherwise
                return a dense Numpy array (default: False).
    Returns:
        array: if sparse=False.
        csr_matrix: if sparse=True.
    """
    def count1(i):
        """Count number of set bits in int or array"""
        i = i - ((i >> 1) & 0x55555555)
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
        return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24

    num_qubits = z.size
    zx_phase = phase + np.sum(x & z)

    dim = 2**num_qubits
    twos_array = 1 << np.arange(num_qubits)
    x_indices = np.array(x).dot(twos_array)
    z_indices = np.array(z).dot(twos_array)

    indptr = np.arange(dim + 1, dtype=np.uint)
    indices = indptr ^ x_indices
    data = (-1)**np.mod(count1(z_indices & indptr), 2)
    if zx_phase:
        data = (-1j)**zx_phase * data

    if sparse:
        # Return sparse matrix
        return csr_matrix((data, indices, indptr),
                          shape=(dim, dim),
                          dtype=complex)

    # Build dense matrix using csr format
    mat = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        mat[i][indices[indptr[i]:indptr[i + 1]]] = data[indptr[i]:indptr[i +
                                                                         1]]
    return mat


# ---------------------------------------------------------------------
# Pauli Evolution by Clifford
# ---------------------------------------------------------------------


def evolve_pauli(pauli, circuit, qargs=None):
    """Update Pauli inplace by applying a Clifford circuit.

    Args:
        pauli (Pauli): the Pauli to update.
        circuit (QuantumCircuit or Instruction): the gate or composite gate to apply.
        qargs (list or None): The qubits to apply gate to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        QiskitError: if input gate cannot be decomposed into Clifford gates.
    """
    if isinstance(circuit, Barrier):
        return pauli

    if qargs is None:
        qargs = list(range(pauli.num_qubits))

    if isinstance(circuit, QuantumCircuit):
        phase = float(circuit.global_phase)
        if phase:
            pauli.phase += coeff_phase_from_complex(np.exp(1j * phase))
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
        return basis_1q[name](pauli, qargs[0])
    if name in basis_2q:
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate.")
        return basis_2q[name](pauli, qargs[0], qargs[1])

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
        evolve_pauli(pauli, instr, new_qubits)
    return pauli


def _evolve_h(pauli, qubit):
    """Evolve by HGate"""
    x = pauli.x[qubit]
    z = pauli.z[qubit]
    pauli.x[qubit] = z
    pauli.z[qubit] = x
    pauli.phase += 2 * (x and z)
    return pauli


def _evolve_s(pauli, qubit):
    """Evolve by SGate"""
    x = pauli.x[qubit]
    pauli.z[qubit] ^= x
    pauli.phase += x
    return pauli


def _evolve_sdg(pauli, qubit):
    """Evolve by SdgGate"""
    x = pauli.x[qubit]
    pauli.z[qubit] ^= x
    pauli.phase -= x
    return pauli


# pylint: disable=unused-argument
def _evolve_i(pauli, qubit):
    """Evolve by IGate"""
    return pauli


def _evolve_x(pauli, qubit):
    """Evolve by XGate"""
    pauli.phase += 2 * pauli.z[qubit]
    return pauli


def _evolve_y(pauli, qubit):
    """Evolve by YGate"""
    pauli.phase += 2 * pauli.x[qubit] + 2 * pauli.z[qubit]
    return pauli


def _evolve_z(pauli, qubit):
    """Evolve by ZGate"""
    pauli.phase += 2 * pauli.x[qubit]
    return pauli


def _evolve_cx(pauli, qctrl, qtrgt):
    """Evolve by CXGate"""
    pauli.x[qtrgt] ^= pauli.x[qctrl]
    pauli.z[qctrl] ^= pauli.z[qtrgt]
    return pauli


def _evolve_cz(pauli, q1, q2):
    """Evolve by CZGate"""
    x1 = pauli.x[q1]
    x2 = pauli.x[q2]
    pauli.z[q1] ^= x1
    pauli.z[q2] ^= x2
    pauli.phase += 2 * (x1 & x2)
    return pauli


def _evolve_cy(pauli, qctrl, qtrgt):
    """Evolve by CYGate"""
    x1 = pauli.x[qctrl]
    x2 = pauli.x[qtrgt]
    z2 = pauli.z[qtrgt]
    pauli.x[qtrgt] ^= x1
    pauli.z[qtrgt] ^= x1
    pauli.z[qctrl] ^= (x2 ^ z2)
    pauli.phase += x1 + 2 * (x1 & x2)
    return pauli


def _evolve_swap(pauli, q1, q2):
    """Evolve by SwapGate"""
    x1 = pauli.x[q1]
    z1 = pauli.z[q1]
    pauli.x[q1] = pauli.x[q2]
    pauli.z[q1] = pauli.z[q2]
    pauli.x[q2] = x1
    pauli.z[q2] = z1
    return pauli
