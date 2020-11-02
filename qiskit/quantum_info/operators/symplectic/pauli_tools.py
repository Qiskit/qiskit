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


def pauli_from_label(label: str, zx_phase: bool = False) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """Return the symplectic representation of Pauli string.

    Args:
        label: the Pauli string label.
        zx_phase: Optional. If True zx-phase convention instead of the group
                  phase convention (See Additional information).

    Returns:
        (z, x, q): the z vector, x vector, and phase integer q for the
                   symplectic representation.

    Raises:
        QiskitError: if Pauli string is not valid.

    Additional Information:
        There are two possible conventions for the returned phase q. The default
        is the group phase defined as :math:`P = (-i)^{q + z.x} Z^z.x^x`.
        If ``zx_phase=True`` the zx_phase convention
        :math:`P = (-i)^{q} Z^z.x^x` is used instead.
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
    z = np.zeros((1, num_qubits), dtype=np.bool)
    x = np.zeros((1, num_qubits), dtype=np.bool)
    phase = np.array([phase], dtype=np.int)
    for i, char in enumerate(pauli):
        if char == 'X':
            x[0, num_qubits - 1 - i] = True
        elif char == 'Z':
            z[0, num_qubits - 1 - i] = True
        elif char == 'Y':
            x[0, num_qubits - 1 - i] = True
            z[0, num_qubits - 1 - i] = True
            if zx_phase:
                phase += 1
    return z, x, phase % 4


def pauli_to_label(z: np.ndarray,
                   x: np.ndarray,
                   phase: int = 0,
                   full_group: bool = True,
                   zx_phase: bool = False,
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
        zx_phase: Optional. If True use zx-phase convention instead of group
                  phase convention.
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
            if zx_phase:
                phase -= 1
    if phase and full_group:
        label = phase_to_coeff_label(phase) + label
    if return_phase:
        return label, phase
    return label


def pauli_to_matrix(z: np.ndarray,
                    x: np.ndarray,
                    phase: int = 0,
                    zx_phase: bool = False,
                    sparse: bool = False
                    ) -> Union[np.ndarray, csr_matrix]:
    """Return the matrix matrix from symplectic representation.

    The Pauli is defined as :math:`P = (-i)^{phase + z.x} * Z^z.x^x`
    where ``array = [x, z]``.

    Args:
        z: The symplectic representation z vector.
        x: The symplectic representation x vector.
        phase: Pauli phase.
        zx_phase: Optional. If True use zx-phase convention instead of group
                  phase convention.
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

    # Convert to zx_phase
    if not zx_phase:
        phase += np.sum(x & z)

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
        mat[i][indices[indptr[i]:indptr[i + 1]]] = data[indptr[i]:indptr[i + 1]]
    return mat
