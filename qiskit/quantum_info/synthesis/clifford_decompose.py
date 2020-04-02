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
Circuit synthesis for the Clifford class.
"""
# pylint: disable=invalid-name

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_z, _append_x, _append_h, _append_s, _append_cx, _append_swap)


def decompose_clifford(clifford):
    """Decompose a Clifford operator into a QuantumCircuit.

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
        # put a 1 one into position by permuting and using Hadamards(i,i)
        _set_qubit_x_true(clifford_cpy, circuit, i)
        # make all entries in row i except ith equal to 0
        # by using phase gate and CNOTS
        _set_row_x_zero(clifford_cpy, circuit, i)
        # treat Zs
        _set_row_z_zero(clifford_cpy, circuit, i)

    for i in range(clifford.num_qubits):
        if clifford_cpy.destabilizer.phase[i]:
            _append_z(clifford_cpy, i)
            circuit.z(i)
        if clifford_cpy.stabilizer.phase[i]:
            _append_x(clifford_cpy, i)
            circuit.x(i)
    # Next we invert the circuit to undo the row reduction and return the
    # result as a gate instruction
    return circuit.inverse()


# ---------------------------------------------------------------------
# Helper functions for decomposition
# ---------------------------------------------------------------------

def _set_qubit_x_true(clifford, circuit, qubit):
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
            _append_swap(clifford, i, qubit)
            circuit.swap(i, qubit)
            return

    # no non-zero element found: need to apply Hadamard somewhere
    for i in range(qubit, clifford.num_qubits):
        if z[i]:
            _append_h(clifford, i)
            circuit.h(i)
            if i != qubit:
                _append_swap(clifford, i, qubit)
                circuit.swap(i, qubit)
            return


def _set_row_x_zero(clifford, circuit, qubit):
    """Set destabilizer.X[qubit, i] to False for all i > qubit.

    This is done by applying CNOTS assumes k<=N and A[k][k]=1
    """
    x = clifford.destabilizer.X[qubit]
    z = clifford.destabilizer.Z[qubit]

    # Check X first
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            _append_cx(clifford, qubit, i)
            circuit.cx(qubit, i)

    # Check whether Zs need to be set to zero:
    if np.any(z[qubit:]):
        if not z[qubit]:
            # to treat Zs: make sure row.Z[k] to True
            _append_s(clifford, qubit)
            circuit.s(qubit)

        # reverse CNOTS
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                _append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)
        # set row.Z[qubit] to False
        _append_s(clifford, qubit)
        circuit.s(qubit)


def _set_row_z_zero(clifford, circuit, qubit):
    """Set stabilizer.Z[qubit, i] to False for all i > qubit.

    Implemented by applying (reverse) CNOTS assumes qubit < num_qubits
    and _set_row_x_zero has been called first
    """

    x = clifford.stabilizer.X[qubit]
    z = clifford.stabilizer.Z[qubit]

    # check whether Zs need to be set to zero:
    if np.any(z[qubit + 1:]):
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                _append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)

    # check whether Xs need to be set to zero:
    if np.any(x[qubit:]):
        _append_h(clifford, qubit)
        circuit.h(qubit)
        for i in range(qubit + 1, clifford.num_qubits):
            if x[i]:
                _append_cx(clifford, qubit, i)
                circuit.cx(qubit, i)
        if z[qubit]:
            _append_s(clifford, qubit)
            circuit.s(qubit)
        _append_h(clifford, qubit)
        circuit.h(qubit)
