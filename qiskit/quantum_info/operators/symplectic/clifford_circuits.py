# -*- coding: utf-8 -*-

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
Circuit methods for Clifford class.
"""
# pylint: disable=invalid-name

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit


# ---------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------

def append_gate(clifford, gate, qargs=None):
    """Update Clifford inplace by applying a Clifford gate.

    Args:
        clifford (Clifford): the Clifford to update.
        gate (Gate or str): the gate or composite gate to apply.
        qargs (list or None): The qubits to apply gate to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        QiskitError: if input gate cannot be decomposed into Clifford gates.
    """
    if qargs is None:
        qargs = list(range(clifford.num_qubits))

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


def decompose_clifford(clifford):
    """Decompose a Clifford into a QuantumCircuit.

    Args:
        clifford (Clifford): a clifford operator.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.
    """
    # Compose a circuit which we will convert to an instruction
    circuit = QuantumCircuit(clifford.num_qubits,
                             name=str(clifford))

    # Make a copy of Clifford as we are going to do row reduction to
    # reduce it to an identity
    clifford_cpy = clifford.copy()

    for i in range(clifford.num_qubits):
        # * make1forXkk(i)

        # put a 1 one into position by permuting and using Hadamards(i,i)
        set_qubit_x_true(clifford_cpy, circuit, i)
        # * .makeXrowzero(i)
        # make all entries in row i except ith equal to 0
        # by using phase gate and CNOTS
        set_row_x_zero(clifford_cpy, circuit, i)
        #  * makeZrowzero(i)
        # treat Zs
        set_row_z_zero(clifford_cpy, circuit, i)

    for i in range(clifford.num_qubits):
        if clifford_cpy.destabilizer.phase[i]:
            append_z(clifford_cpy, i)
            circuit.z(i)
        if clifford_cpy.stabilizer.phase[i]:
            append_x(clifford_cpy, i)
            circuit.x(i)
    # Next we invert the circuit to undo the row reduction and return the
    # result as a gate instruction
    return circuit.inverse()


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
    # pylint: disable=unused-argument
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
    clifford.table.phase ^= x0 & x1 & (z0 ^ z1)
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


# ---------------------------------------------------------------------
# Helper functions for decomposition
# ---------------------------------------------------------------------

def set_qubit_x_true(clifford, circuit, qubit):
    """Set destabilizer.X[qubit, qubit] to be True.

    This is done by permuting columns l > qubit or if necessary applying
    a Hadamard
    """
    x = clifford.destabilizer.X[qubit]
    z = clifford.destabilizer.Z[qubit]

    if x[qubit]:
        return

    # Try to find non-zero element
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            append_swap(clifford, i, qubit)
            circuit.swap(i, qubit)
            return

    # no non-zero element found: need to apply Hadamard somewhere
    for i in range(qubit, clifford.num_qubits):
        if z[i]:
            append_h(clifford, i)
            circuit.h(i)
            if i != qubit:
                append_swap(clifford, i, qubit)
                circuit.swap(i, qubit)
            return


def set_row_x_zero(clifford, circuit, qubit):
    """Set destabilizer.X[qubit, i] to False for all i > qubit.

    This is done by applying CNOTS assumes k<=N and A[k][k]=1
    """
    x = clifford.destabilizer.X[qubit]
    z = clifford.destabilizer.Z[qubit]

    # Check X first
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            append_cx(clifford, qubit, i)
            circuit.cx(qubit, i)

    # Check whether Zs need to be set to zero:
    if np.any(z[qubit:]):
        if not z[qubit]:
            # to treat Zs: make sure row.Z[k] to True
            append_s(clifford, qubit)
            circuit.s(qubit)

        # reverse CNOTS
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)
        # set row.Z[qubit] to False
        append_s(clifford, qubit)
        circuit.s(qubit)


def set_row_z_zero(clifford, circuit, qubit):
    """Set stabilizer.Z[qubit, i] to False for all i > qubit.

    Implemented by applying (reverse) CNOTS assumes qubit < num_qubits
    and set_row_x_zero has been called first
    """

    x = clifford.stabilizer.X[qubit]
    z = clifford.stabilizer.Z[qubit]

    # check whether Zs need to be set to zero:
    if np.any(z[qubit + 1:]):
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)

    # check whether Xs need to be set to zero:
    if np.any(x[qubit:]):
        append_h(clifford, qubit)
        circuit.h(qubit)
        for i in range(qubit + 1, clifford.num_qubits):
            if x[i]:
                append_cx(clifford, qubit, i)
                circuit.cx(qubit, i)
        if z[qubit]:
            append_s(clifford, qubit)
            circuit.s(qubit)
        append_h(clifford, qubit)
        circuit.h(qubit)
