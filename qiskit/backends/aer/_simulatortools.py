# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Contains functions used by the simulators.

Functions
    index2 -- Takes a bitstring k and inserts bits b1 as the i1th bit
    and b2 as the i2th bit
"""

from string import ascii_uppercase, ascii_lowercase
import numpy as np
from sympy import Matrix, pi, E, I, cos, sin, N, sympify

from qiskit import QISKitError


def index1(b, i, k):
    """Magic index1 function.

    Takes a bitstring k and inserts bit b as the ith bit,
    shifting bits >= i over to make room.
    """
    retval = k
    lowbits = k & ((1 << i) - 1)  # get the low i bits

    retval >>= i
    retval <<= 1

    retval |= b

    retval <<= i
    retval |= lowbits

    return retval


def index2(b1, i1, b2, i2, k):
    """Magic index1 function.

    Takes a bitstring k and inserts bits b1 as the i1th bit
    and b2 as the i2th bit
    """
    assert i1 != i2

    if i1 > i2:
        # insert as (i1-1)th bit, will be shifted left 1 by next line
        retval = index1(b1, i1-1, k)
        retval = index1(b2, i2, retval)
    else:  # i2>i1
        # insert as (i2-1)th bit, will be shifted left 1 by next line
        retval = index1(b2, i2-1, k)
        retval = index1(b1, i1, retval)
    return retval


def single_gate_params(gate, params=None):
    """Apply a single qubit gate to the qubit.

    Args:
        gate(str): the single qubit gate name
        params(list): the operation parameters op['params']
    Returns:
        tuple: a tuple of U gate parameters (theta, phi, lam)
    Raises:
        QISKitError: if the gate name is not valid
    """
    if gate == 'U' or gate == 'u3':
        return params[0], params[1], params[2]
    elif gate == 'u2':
        return np.pi/2, params[0], params[1]
    elif gate == 'u1':
        return 0, 0, params[0]
    elif gate == 'id':
        return 0, 0, 0
    raise QISKitError('Gate is not among the valid types: %s' % gate)


def single_gate_matrix(gate, params=None):
    """Get the matrix for a single qubit.

    Args:
        gate(str): the single qubit gate name
        params(list): the operation parameters op['params']
    Returns:
        array: A numpy array representing the matrix
    """

    # Converting sym to floats improves the performance of the simulator 10x.
    # This a is a probable a FIXME since it might show bugs in the simulator.
    (theta, phi, lam) = map(float, single_gate_params(gate, params))

    return np.array([[np.cos(theta/2),
                      -np.exp(1j*lam)*np.sin(theta/2)],
                     [np.exp(1j*phi)*np.sin(theta/2),
                      np.exp(1j*phi+1j*lam)*np.cos(theta/2)]])


def einsum_matmul_index(gate_indices, number_of_qubits):
    """Return the index string for Numpy.eignsum matrix multiplication.

    The returned indices are to perform a matrix multiplication A.B where
    the matrix A is an M-qubit matrix, matrix B is an N-qubit matrix, and
    M <= N, and identity matrices are implied on the subsystems where A has no
    support on B.

    Args:
        gate_indices (list[int]): the indices of the right matrix subsystems
                                   to contract with the left matrix.
        number_of_qubits (int): the total number of qubits for the right matrix.

    Returns:
        str: An indices string for the Numpy.einsum function.

    Raises:
        QISKitError: if the total number of qubits plus the number of
        contracted indices is greater than 26.
    """

    # Since we use ASCII alphabet for einsum index labels we are limited
    # to 26 total free left (lowercase) and 26 right (uppercase) indexes.
    # The rank of the contracted tensor reduces this as we need to use that
    # many characters for the contracted indices
    if len(gate_indices) + number_of_qubits > 26:
        raise QISKitError("Total number of free indexes limited to 26")

    # Right indices for the N-qubit input and output tensor
    idx_right = ascii_uppercase[:number_of_qubits]

    # Left ndicies for N-qubit input tensor
    idx_left_in = ascii_lowercase[:number_of_qubits]

    # Left indices for the N-qubit output tensor
    idx_left_out = list(idx_left_in)

    # Left and right indices for the M-qubit multiplying tensor
    mat_left = ""
    mat_right = ""

    # Update left indices for mat and output
    for pos, idx in enumerate(reversed(gate_indices)):
        mat_left += ascii_lowercase[-1 - pos]
        mat_right += idx_left_in[-1 - idx]
        idx_left_out[-1 - idx] = ascii_lowercase[-1 - pos]
    idx_left_out = "".join(idx_left_out)

    # Combine indices into matrix multiplication string format
    # for numpy.einsum function
    return "{mat_l}{mat_r}, ".format(mat_l=mat_left, mat_r=mat_right) + \
           "{tens_lin}{tens_r}->{tens_lout}{tens_r}".format(tens_lin=idx_left_in,
                                                            tens_lout=idx_left_out,
                                                            tens_r=idx_right)


# Functions used by the sympy simulators.

def regulate(theta):
    """
    Return the regulated symbolic representation of `theta`::
        * if it has a representation close enough to `pi` transformations,
            return that representation (for example, `3.14` -> `sympy.pi`).
        * otherwise, return a sympified representation of theta (for example,
            `1.23` ->  `sympy.Float(1.23)`).

    See also `UGateGeneric`.

    Args:
        theta (float or sympy.Basic): the float value (e.g., 3.14) or the
            symbolic value (e.g., pi)

    Returns:
        sympy.Basic: the sympy-regulated representation of `theta`
    """
    error_margin = 0.01
    targets = [pi, pi/2, pi * 2, pi / 4]

    for t in targets:
        if abs(N(theta - t)) < error_margin:
            return t

    return sympify(theta)


def compute_ugate_matrix(parameters):
    """Compute the matrix associated with a parameterized U gate.

    Args:
        parameters (list[float]): parameters carried by the U gate
    Returns:
        sympy.Matrix: the matrix associated with a parameterized U gate
    """
    theta = regulate(parameters[0])
    phi = regulate(parameters[1])
    lamb = regulate(parameters[2])

    left_up = cos(theta/2)
    right_up = (-E**(I*lamb)) * sin(theta/2)
    left_down = (E**(I*phi)) * sin(theta/2)
    right_down = (E**(I*(phi + lamb))) * cos(theta/2)

    return Matrix([[left_up, right_up], [left_down, right_down]])
