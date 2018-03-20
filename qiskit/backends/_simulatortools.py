# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Contains functions used by the simulators.

Functions
    index2 -- Takes a bitstring k and inserts bits b1 as the i1th bit
    and b2 as the i2th bit

    enlarge_single_opt(opt, qubit, number_of_qubits) -- takes a single qubit
    operator opt to a opterator on n qubits

    enlarge_two_opt(opt, q0, q1, number_of_qubits) -- takes a two-qubit
    operator opt to a opterator on n qubits

"""
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


def enlarge_single_opt(opt, qubit, number_of_qubits):
    """Enlarge single operator to n qubits.

    It is exponential in the number of qubits.

    Args:
        opt (array): the single-qubit opt.
        qubit (int): the qubit to apply it on counts from 0 and order
            is q_{n-1} ... otimes q_1 otimes q_0.
        number_of_qubits (int): the number of qubits in the system.

    Returns:
        array: enlarge single operator to n qubits
    """
    temp_1 = np.identity(2**(number_of_qubits-qubit-1), dtype=complex)
    temp_2 = np.identity(2**qubit, dtype=complex)
    enlarge_opt = np.kron(temp_1, np.kron(opt, temp_2))
    return enlarge_opt


def enlarge_two_opt(opt, q0, q1, num):
    """Enlarge two-qubit operator to n qubits.

    It is exponential in the number of qubits.
    opt is the two-qubit gate
    q0 is the first qubit (control) counts from 0
    q1 is the second qubit (target)
    returns a complex numpy array
    number_of_qubits is the number of qubits in the system.
    """
    enlarge_opt = np.zeros([1 << (num), 1 << (num)])
    for i in range(1 << (num-2)):
        for j in range(2):
            for k in range(2):
                for jj in range(2):
                    for kk in range(2):
                        enlarge_opt[index2(j, q0, k, q1, i),
                                    index2(jj, q0, kk, q1, i)] = opt[j+2*k,
                                                                     jj+2*kk]
    return enlarge_opt


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
