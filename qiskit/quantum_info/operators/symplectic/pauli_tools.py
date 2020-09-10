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


def pauli_from_label(label: str) -> Tuple[np.ndarray, int]:
    """Return the symplectic representation of Pauli string."""
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
    arr = np.zeros(2 * num_qubits, dtype=np.bool)
    xs = arr[0:num_qubits]
    zs = arr[num_qubits:2 * num_qubits]
    for i, char in enumerate(pauli):
        if char == 'X':
            xs[num_qubits - 1 - i] = True
        elif char == 'Z':
            zs[num_qubits - 1 - i] = True
        elif char == 'Y':
            xs[num_qubits - 1 - i] = True
            zs[num_qubits - 1 - i] = True
            phase += 1
    return arr, phase % 4


def pauli_to_label(array: np.ndarray,
                   phase: int = 0,
                   full_group: bool = True,
                   return_phase: bool = False) -> Union[str, Tuple[str, int]]:
    """Return the label string for a Pauli.

    The pauli operator is given by P = (-1j) ** phase * array

    Args:
        array: The symplectic :math:`[X, Z]` array of a Pauli.
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
                         the phase ``q`` for the coefficient :math:`(-i)^q` for the
                         label from the full Pauli group.
    """
    num_qubits = array.size // 2
    x = array[0:num_qubits]
    z = array[num_qubits:2 * num_qubits]
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
            phase -= 1
    if full_group:
        label = phase_to_coeff_label(phase) + label
    if return_phase:
        return label, phase
    return label


def pauli_to_matrix(pauli: np.ndarray, phase: int = 0,
                    sparse=False) -> Union[np.ndarray, csr_matrix]:
    """Return the matrix matrix from symplectic representation.

    Args:
        pauli (array): symplectic Pauli vector.
        phase (int): Pauli phase.
        sparse (bool): if True return a sparse CSR matrix, otherwise
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

    symp = np.asarray(pauli, dtype=np.bool)
    num_qubits = symp.size // 2
    x = symp[0:num_qubits]
    z = symp[num_qubits:2 * num_qubits]

    dim = 2**num_qubits
    twos_array = 1 << np.arange(num_qubits)
    x_indices = np.array(x).dot(twos_array)
    z_indices = np.array(z).dot(twos_array)

    indptr = np.arange(dim + 1, dtype=np.uint)
    indices = indptr ^ x_indices
    data = (-1)**np.mod(count1(z_indices & indptr), 2)
    if phase:
        data = (-1j)**phase * data

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
    x = pauli.X[qubit]
    z = pauli.Z[qubit]
    pauli.X[qubit] = z
    pauli.Z[qubit] = x
    pauli.phase += 2 * (x and z)
    return pauli


def _evolve_s(pauli, qubit):
    """Evolve by SGate"""
    x = pauli.X[qubit]
    pauli.Z[qubit] ^= x
    pauli.phase += x
    return pauli


def _evolve_sdg(pauli, qubit):
    """Evolve by SdgGate"""
    x = pauli.X[qubit]
    pauli.Z[qubit] ^= x
    pauli.phase -= x
    return pauli


# pylint: disable=unused-argument
def _evolve_i(pauli, qubit):
    """Evolve by IGate"""
    return pauli


def _evolve_x(pauli, qubit):
    """Evolve by XGate"""
    pauli.phase += 2 * pauli.Z[qubit]
    return pauli


def _evolve_y(pauli, qubit):
    """Evolve by YGate"""
    pauli.phase += 2 * pauli.X[qubit] + 2 * pauli.Z[qubit]
    return pauli


def _evolve_z(pauli, qubit):
    """Evolve by ZGate"""
    pauli.phase += 2 * pauli.X[qubit]
    return pauli


def _evolve_cx(pauli, qctrl, qtrgt):
    """Evolve by CXGate"""
    pauli.X[qtrgt] ^= pauli.X[qctrl]
    pauli.Z[qctrl] ^= pauli.Z[qtrgt]
    return pauli


def _evolve_cz(pauli, q1, q2):
    """Evolve by CZGate"""
    x1 = pauli.X[q1]
    x2 = pauli.X[q2]
    pauli.Z[q1] ^= x1
    pauli.Z[q2] ^= x2
    pauli.phase += 2 * (x1 & x2)
    return pauli


def _evolve_cy(pauli, qctrl, qtrgt):
    """Evolve by CYGate"""
    x1 = pauli.X[qctrl]
    x2 = pauli.X[qtrgt]
    z2 = pauli.Z[qtrgt]
    pauli.X[qtrgt] ^= x1
    pauli.Z[qtrgt] ^= x1
    pauli.Z[qctrl] ^= (x2 ^ z2)
    pauli.phase += x1 + 2 * (x1 & x2)
    return pauli


def _evolve_swap(pauli, q1, q2):
    """Evolve by SwapGate"""
    x1 = pauli.X[q1]
    z1 = pauli.Z[q1]
    pauli.X[q1] = pauli.X[q2]
    pauli.Z[q1] = pauli.Z[q2]
    pauli.X[q2] = x1
    pauli.Z[q2] = z1
    return pauli
