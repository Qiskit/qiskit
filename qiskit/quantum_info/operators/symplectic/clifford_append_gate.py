# -*- coding: utf-8 -*-

# Copyright 2017, 2020 BM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Clifford class gate update utility function
"""
# pylint: disable=invalid-name

from qiskit import QiskitError


# ---------------------------------------------------------------------
# Apply Clifford Gates
# ---------------------------------------------------------------------

def append_gate(clifford, gate, qargs=None):
    """Update Clifford inplace by applying a Clifford gate.

    Args:
        clifford (Clifford): the Clifford to update.
        gate (Gate or str): the gate or composite gate to apply.
        qargs (list or None): The qubits to apply gate to.

    Returns:
        Clifford: the updated Clifford.
    """
    if qargs is None:
        qargs = list(range(clifford.n_qubits))

    # Basis Clifford Gates
    basis_1q = {
        'i': append_i, 'id': append_i, 'iden': append_i,
        'x': append_x, 'y': append_y, 'z': append_z, 'h': append_h,
        's': append_s, 'sdg': append_sdg, 'sinv': append_sdg,
        'v': append_v, 'w': append_w
    }
    basis_2q = {
        'cx': append_cx, 'cz': append_cz, 'swap': append_swap
    }

    # Non-clifford gates
    non_clifford = ['t', 'tdg', 'ccx', 'ccz']

    if isinstance(gate, str):
        # Check if gate is a valid Clifford basis gate string
        if gate not in basis_1q and gate not in basis_2q:
            raise QiskitError("Invalid Clifford gate name string {}".format(gate))
        name = gate
    else:
        # Assume gate is an Instruction
        name = gate.name

    # Apply gate if it is a Clifford basis gate
    if name in non_clifford:
        raise QiskitError(
            "Cannot update Clifford with non-Clifford gate {}".format(name))
    if name in basis_1q:
        if len(qargs) != 1:
            raise QiskitError("Invalid qubits for 1-qubit gate.")
        return basis_1q[name](clifford, qargs[0])
    if name in basis_2q:
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate.")
        return basis_2q[name](clifford, qargs[0], qargs[1])

    # If not a Clifford basis gate we try to unroll the gate,
    # raising an exception if unrolling reaches a non-Clifford gate.
    # TODO: We could check for also check u3 params to see if they
    # are a single qubit Clifford gate rather than raise an exception.
    if gate.definition is None:
        raise QiskitError('Cannot apply Instruction: {}'.format(gate.name))
    for instr, qregs, cregs in gate.definition:
        if cregs:
            raise QiskitError(
                'Cannot apply Instruction with classical registers: {}'.format(
                    instr.name))
        # Get the integer position of the flat register
        new_qubits = [qargs[tup.index] for tup in qregs]
        append_gate(clifford, instr, new_qubits)
    return clifford


# ---------------------------------------------------------------------
# Helper functions for applying basis gates
# ---------------------------------------------------------------------

def append_i(clifford, qubit):
    """Apply an I gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    return clifford


def append_x(clifford, qubit):
    """Apply an X gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.table.phase ^= clifford.table.Z[:, qubit]
    return clifford


def append_y(clifford, qubit):
    """Apply a Y gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]
    clifford.table.phase ^= x ^ z
    return clifford


def append_z(clifford, qubit):
    """Apply an Z gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.table.phase ^= clifford.table.X[:, qubit]
    return clifford


def append_h(clifford, qubit):
    """Apply a H gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]
    clifford.table.phase ^= x & z
    tmp = x.copy()
    x[:] = z
    z[:] = tmp
    return clifford


def append_s(clifford, qubit):
    """Apply an S gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]

    clifford.table.phase ^= x & z
    z ^= x
    return clifford


def append_sdg(clifford, qubit):
    """Apply an Sdg gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]
    clifford.table.phase ^= x & ~z
    z ^= x
    return clifford


def append_v(clifford, qubit):
    """Apply a V gate to a Clifford.

    This is equivalent to an Sdg gate followed by a H gate.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]
    clifford.table.phase ^= False
    tmp = x.copy()
    x ^= z
    z[:] = tmp
    return clifford


def append_w(clifford, qubit):
    """Apply a W gate to a Clifford.

    This is equivalent to two V gates.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]
    clifford.table.phase ^= False
    tmp = z.copy()
    z ^= x
    x[:] = tmp
    return clifford


def append_cx(clifford, control, target):
    """Apply a CX gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x0 = clifford.table.X[:, control]
    z0 = clifford.table.Z[:, control]
    x1 = clifford.table.X[:, target]
    z1 = clifford.table.Z[:, target]
    clifford.table.phase ^= (x1 ^ z0 ^ True) & z1 & x0
    x1 ^= x0
    z0 ^= z1
    return clifford


def append_cz(clifford, control, target):
    """Apply a CZ gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x0 = clifford.table.X[:, control]
    z0 = clifford.table.Z[:, control]
    x1 = clifford.table.X[:, target]
    z1 = clifford.table.Z[:, target]
    clifford.table.phase ^= (z1 ^ z0 ^ True) & x1 & x0
    z1 ^= x0
    z0 ^= x1
    return clifford


def append_swap(clifford, qubit0, qubit1):
    """Apply a Swap gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.table.X[:, [qubit0, qubit1]] = clifford.table.X[:, [qubit1, qubit0]]
    clifford.table.Z[:, [qubit0, qubit1]] = clifford.table.Z[:, [qubit1, qubit0]]
    return clifford
