# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Contains helper functions used by the simulators."""

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
    """Magic index2 function.

    Takes a bitstring k and inserts bits b1 as the i1th bit
    and b2 as the i2th bit
    """
    if i1 == i2:
        raise QISKitError("can't insert two bits to same place")

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

    # Left indices for N-qubit input tensor
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


# Functions used by the noise simulators.
def cx_error_matrix(cal_error, zz_error):
    """
    Return the coherent error matrix for CR error model of a CNOT gate.
    Args:
        cal_error (double): calibration error of rotation
        zz_error (double): ZZ interaction term error
    Returns:
        numpy.ndarray: A coherent error matrix U_error for the CNOT gate.
    Details:
    The ideal cross-resonsance (CR) gate corresponds to a 2-qubit rotation
        U_CR_ideal = exp(-1j * (pi/2) * XZ/2)
    where qubit-0 is the control, and qubit-1 is the target. This can be
    converted to a CNOT gate by single-qubit rotations::
        U_CX = U_L * U_CR_ideal * U_R
    The noisy rotation is implemented as
        U_CR_noise = exp(-1j * (pi/2 + cal_error) * (XZ + zz_error ZZ)/2)
    The retured error matrix is given by
        U_error = U_L * U_CR_noise * U_R * U_CX^dagger
    """
    # pylint: disable=invalid-name
    if cal_error == 0 and zz_error == 0:
        return np.eye(4)
    cx_ideal = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0]])
    b = np.sqrt(1.0 + zz_error * zz_error)
    a = b * (np.pi / 2.0 + cal_error) / 2.0
    sp = (1.0 + 1j * zz_error) * np.sin(a) / b
    sm = (1.0 - 1j * zz_error) * np.sin(a) / b
    c = np.cos(a)
    cx_noise = np.array([[c + sm, 0, -1j * (c - sm), 0],
                         [0, 1j * (c - sm), 0, c + sm],
                         [-1j * (c - sp), 0, c + sp, 0],
                         [0, c + sp, 0, 1j * (c - sp)]]) / np.sqrt(2)
    return cx_noise.dot(cx_ideal.conj().T)


def x90_error_matrix(cal_error, detuning_error):
    """
    Return the coherent error matrix for a X90 rotation gate.
    Args:
        cal_error (double): calibration error of rotation
        detuning_error (double): detuning amount for rotation axis error
    Returns:
        numpy.ndarray: A coherent error matrix U_error for the X90 gate.
    Details:
    The ideal X90 rotation is a pi/2 rotation about the X-axis:
        U_X90_ideal = exp(-1j (pi/2) X/2)
    The noisy rotation is implemented as
        U_X90_noise = exp(-1j (pi/2 + cal_error) (cos(d) X + sin(d) Y)/2)
    where d is the detuning_error.
    The retured error matrix is given by
        U_error = U_X90_noise * U_X90_ideal^dagger
    """
    # pylint: disable=invalid-name
    if cal_error == 0 and detuning_error == 0:
        return np.eye(2)
    else:
        x90_ideal = np.array([[1., -1.j], [-1.j, 1]]) / np.sqrt(2)
        c = np.cos(0.5 * cal_error)
        s = np.sin(0.5 * cal_error)
        gamma = np.exp(-1j * detuning_error)
        x90_noise = np.array([[c - s, -1j * (c + s) * gamma],
                              [-1j * (c + s) * np.conj(gamma), c - s]]) / np.sqrt(2)
    return x90_noise.dot(x90_ideal.conj().T)


def _generate_coherent_error_matrix(config):
    """
    Generate U_error matrix for CX and X90 gates.
    Args:
        config (dict): the config of a qobj circuit
    This parses the config for the following noise parameter keys and returns a
    coherent error matrix for simulation coherent noise::
        * 'CX' gate: 'calibration_error', 'zz_error'
        * 'X90' gate: 'calibration_error', 'detuning_error'
    """
    # pylint: disable=invalid-name
    if 'noise_params' in config:
        # Check for CR coherent error parameters
        if 'CX' in config['noise_params']:
            noise_cx = config['noise_params']['CX']
            cal_error = noise_cx.pop('calibration_error', 0)
            zz_error = noise_cx.pop('zz_error', 0)
            # Add to current coherent error matrix
            if not cal_error == 0 or not zz_error == 0:
                u_error = noise_cx.get('U_error', np.eye(4))
                u_error = u_error.dot(cx_error_matrix(cal_error, zz_error))
                config['noise_params']['CX']['U_error'] = u_error
        # Check for X90 coherent error parameters
        if 'X90' in config['noise_params']:
            noise_x90 = config['noise_params']['X90']
            cal_error = noise_x90.pop('calibration_error', 0)
            detuning_error = noise_x90.pop('detuning_error', 0)
            # Add to current coherent error matrix
            if not cal_error == 0 or not detuning_error == 0:
                u_error = noise_x90.get('U_error', np.eye(2))
                u_error = u_error.dot(x90_error_matrix(cal_error,
                                                       detuning_error))
                config['noise_params']['X90']['U_error'] = u_error
