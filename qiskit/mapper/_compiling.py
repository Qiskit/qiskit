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

"""
Methods to assist with compiling tasks.
"""
import math

import numpy as np
from scipy.linalg import expm

from ._mappererror import MapperError


def euler_angles_1q(unitary_matrix):
    """Compute Euler angles for a single-qubit gate.

    Find angles (theta, phi, lambda) such that
    unitary_matrix = phase * Rz(phi) * Ry(theta) * Rz(lambda)

    Return (theta, phi, lambda, "U(theta,phi,lambda)"). The last
    element of the tuple is the OpenQASM gate name with parameter
    values substituted.
    """
    small = 1e-10
    if unitary_matrix.shape != (2, 2):
        raise MapperError("compiling.euler_angles_1q expected 2x2 matrix")
    phase = np.linalg.det(unitary_matrix)**(-1.0/2.0)
    U = phase * unitary_matrix  # U in SU(2)
    # OpenQASM SU(2) parameterization:
    # U[0, 0] = exp(-i(phi+lambda)/2) * cos(theta/2)
    # U[0, 1] = -exp(-i(phi-lambda)/2) * sin(theta/2)
    # U[1, 0] = exp(i(phi-lambda)/2) * sin(theta/2)
    # U[1, 1] = exp(i(phi+lambda)/2) * cos(theta/2)
    # Find theta
    if abs(U[0, 0]) > small:
        theta = 2 * math.acos(abs(U[0, 0]))
    else:
        theta = 2 * math.asin(abs(U[1, 0]))
    # Find phi and lambda
    phase11 = 0.0
    phase10 = 0.0
    if abs(math.cos(theta/2.0)) > small:
        phase11 = U[1, 1] / math.cos(theta/2.0)
    if abs(math.sin(theta/2.0)) > small:
        phase10 = U[1, 0] / math.sin(theta/2.0)
    phiplambda = 2 * math.atan2(np.imag(phase11), np.real(phase11))
    phimlambda = 2 * math.atan2(np.imag(phase10), np.real(phase10))
    phi = 0.0
    if abs(U[0, 0]) > small and abs(U[1, 0]) > small:
        phi = (phiplambda + phimlambda) / 2.0
        lamb = (phiplambda - phimlambda) / 2.0
    else:
        if abs(U[0, 0]) < small:
            lamb = -phimlambda
        else:
            lamb = phiplambda
    # Check the solution
    Rzphi = np.array([[np.exp(-1j*phi/2.0), 0],
                      [0, np.exp(1j*phi/2.0)]], dtype=complex)
    Rytheta = np.array([[np.cos(theta/2.0), -np.sin(theta/2.0)],
                        [np.sin(theta/2.0), np.cos(theta/2.0)]], dtype=complex)
    Rzlambda = np.array([[np.exp(-1j*lamb/2.0), 0],
                         [0, np.exp(1j*lamb/2.0)]], dtype=complex)
    V = np.dot(Rzphi, np.dot(Rytheta, Rzlambda))
    if np.linalg.norm(V - U) > small:
        raise MapperError("compiling.euler_angles_1q incorrect result")
    return theta, phi, lamb, "U(%.15f,%.15f,%.15f)" % (theta, phi, lamb)


def simplify_U(theta, phi, lam):
    """Return the gate u1, u2, or u3 implementing U with the fewest pulses.

    U(theta, phi, lam) is the input gate.

    The returned gate implements U exactly, not up to a global phase.

    Return (gate_string, params, "OpenQASM string") where gate_string is one of
    "u1", "u2", "u3", "id" and params is a 3-tuple of parameter values. The
    OpenQASM string is the name of the gate with parameters substituted.
    """
    epsilon = 1e-13
    name = "u3"
    params = (theta, phi, lam)
    qasm = "u3(%.15f,%.15f,%.15f)" % params
    # Y rotation is 0 mod 2*pi, so the gate is a u1
    if abs(params[0] % (2.0 * math.pi)) < epsilon:
        name = "u1"
        params = (0.0, 0.0, params[1] + params[2] + params[0])
        qasm = "u1(%.15f)" % params[2]
    # Y rotation is pi/2 or -pi/2 mod 2*pi, so the gate is a u2
    if name == "u3":
        # theta = pi/2 + 2*k*pi
        if abs((params[0] - math.pi / 2) % (2.0 * math.pi)) < epsilon:
            name = "u2"
            params = (math.pi / 2, params[1],
                      params[2] + (params[0] - math.pi / 2))
            qasm = "u2(%.15f,%.15f)" % (params[1], params[2])
        # theta = -pi/2 + 2*k*pi
        if abs((params[0] + math.pi / 2) % (2.0 * math.pi)) < epsilon:
            name = "u2"
            params = (math.pi / 2, params[1] + math.pi,
                      params[2] - math.pi + (params[0] + math.pi / 2))
            qasm = "u2(%.15f,%.15f)" % (params[1], params[2])
    # u1 and lambda is 0 mod 4*pi so gate is nop
    if name == "u1" and abs(params[2] % (4.0 * math.pi)) < epsilon:
        name = "id"
        params = (0.0, 0.0, 0.0)
        qasm = "id"
    return name, params, qasm


def rz_array(theta):
    """Return numpy array for Rz(theta).

    Rz(theta) = diag(exp(-i*theta/2),exp(i*theta/2))
    """
    return np.array([[np.exp(-1j*theta/2.0), 0],
                     [0, np.exp(1j*theta/2.0)]], dtype=complex)


def ry_array(theta):
    """Return numpy array for Ry(theta).

    Ry(theta) = [[cos(theta/2), -sin(theta/2)],
                 [sin(theta/2),  cos(theta/2)]]
    """
    return np.array([[math.cos(theta/2.0), -math.sin(theta/2.0)],
                     [math.sin(theta/2.0), math.cos(theta/2.0)]],
                    dtype=complex)


def two_qubit_kak(unitary_matrix):
    """Decompose a two-qubit gate over CNOT + SU(2) using the KAK decomposition.

    Based on MATLAB implementation by David Gosset.

    Computes a sequence of 10 single and two qubit gates, including 3 CNOTs,
    which multiply to U, including global phase. Uses Vatan and Williams
    optimal two-qubit circuit (quant-ph/0308006v3). The decomposition algorithm
    which achieves this is explained well in Drury and Love, 0806.4015.

    unitary_matrix = numpy 4x4 unitary matrix
    """
    if unitary_matrix.shape != (4, 4):
        raise MapperError("compiling.two_qubit_kak expected 4x4 matrix")
    phase = np.linalg.det(unitary_matrix)**(-1.0/4.0)
    # Make it in SU(4), correct phase at the end
    U = phase * unitary_matrix
    # B changes to the Bell basis
    B = (1.0/math.sqrt(2)) * np.array([[1, 1j, 0, 0],
                                       [0, 0, 1j, 1],
                                       [0, 0, 1j, -1],
                                       [1, -1j, 0, 0]], dtype=complex)
    # U' = Bdag . U . B
    Uprime = np.dot(np.transpose(B.conjugate()), np.dot(U, B))
    # M^2 = trans(U') . U'
    M2 = np.dot(np.transpose(Uprime), Uprime)
    # Diagonalize M2
    # Must use diagonalization routine which finds a real orthogonal matrix P
    # when M2 is real.
    D, P = np.linalg.eig(M2)
    # If det(P) == -1, apply a swap to make P in SO(4)
    if abs(np.linalg.det(P)+1) < 1e-5:
        swap = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
        P = np.dot(P, swap)
        D = np.diag(np.dot(swap, np.dot(np.diag(D), swap)))
    Q = np.diag(np.sqrt(D))  # array from elementwise sqrt
    # Want to take square root so that Q has determinant 1
    if abs(np.linalg.det(Q)+1) < 1e-5:
        Q[0, 0] = -Q[0, 0]
    Kprime = np.dot(Uprime, np.dot(P, np.dot(np.linalg.inv(Q),
                                             np.transpose(P))))
    K1 = np.dot(B, np.dot(Kprime, np.dot(P, np.transpose(B.conjugate()))))
    A = np.dot(B, np.dot(Q, np.transpose(B.conjugate())))
    K2 = np.dot(B, np.dot(np.transpose(P), np.transpose(B.conjugate())))
    KAK = np.dot(K1, np.dot(A, K2))
    if np.linalg.norm(KAK - U, 2) > 1e-6:
        raise MapperError("compiling.two_qubit_kak: " +
                          "unknown error in KAK decomposition")
    # Compute parameters alpha, beta, gamma so that
    # A = exp(i * (alpha * XX + beta * YY + gamma * ZZ))
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    xx = np.kron(x, x)
    yy = np.kron(y, y)
    zz = np.kron(z, z)
    alpha = math.atan(np.trace(np.imag(np.dot(A, xx)))/np.trace(np.real(A)))
    beta = math.atan(np.trace(np.imag(np.dot(A, yy)))/np.trace(np.real(A)))
    gamma = math.atan(np.trace(np.imag(np.dot(A, zz)))/np.trace(np.real(A)))
    # K1 = kron(U1, U2) and K2 = kron(V1, V2)
    # Find the matrices U1, U2, V1, V2
    L = K1[0:2, 0:2]
    if np.linalg.norm(L) < 1e-9:
        L = K1[0:2, 2:4]
        if np.linalg.norm(L) < 1e-9:
            L = K1[2:4, 2:4]
    Q = np.dot(L, np.transpose(L.conjugate()))
    U2 = L / np.sqrt(Q[0, 0])
    R = np.dot(K1, np.kron(np.identity(2), np.transpose(U2.conjugate())))
    U1 = np.array([[0, 0], [0, 0]], dtype=complex)
    U1[0, 0] = R[0, 0]
    U1[0, 1] = R[0, 2]
    U1[1, 0] = R[2, 0]
    U1[1, 1] = R[2, 2]
    L = K2[0:2, 0:2]
    if np.linalg.norm(L) < 1e-9:
        L = K2[0:2, 2:4]
        if np.linalg.norm(L) < 1e-9:
            L = K2[2:4, 2:4]
    Q = np.dot(L, np.transpose(L.conjugate()))
    V2 = L / np.sqrt(Q[0, 0])
    R = np.dot(K2, np.kron(np.identity(2), np.transpose(V2.conjugate())))
    V1 = np.array([[0, 0], [0, 0]], dtype=complex)
    V1[0, 0] = R[0, 0]
    V1[0, 1] = R[0, 2]
    V1[1, 0] = R[2, 0]
    V1[1, 1] = R[2, 2]
    if np.linalg.norm(np.kron(U1, U2) - K1) > 1e-4 or \
       np.linalg.norm(np.kron(V1, V2) - K2) > 1e-4:
        raise MapperError("compiling.two_qubit_kak: " +
                          "error in SU(2) x SU(2) part")
    test = expm(1j*(alpha * xx + beta * yy + gamma * zz))
    if np.linalg.norm(A - test) > 1e-4:
        raise MapperError("compiling.two_qubit_kak: " +
                          "error in A part")
    # Circuit that implements K1 * A * K2 (up to phase), using
    # Vatan and Williams Fig. 6 of quant-ph/0308006v3
    # Include prefix and suffix single-qubit gates into U2, V1 respectively.
    V2 = np.dot(np.array([[np.exp(1j*np.pi/4), 0],
                          [0, np.exp(-1j*np.pi/4)]], dtype=complex), V2)
    U1 = np.dot(U1, np.array([[np.exp(-1j*np.pi/4), 0],
                              [0, np.exp(1j*np.pi/4)]], dtype=complex))
    # Corrects global phase: exp(ipi/4)*phase'
    U1 = np.dot(U1, np.array([[np.exp(1j*np.pi/4), 0],
                              [0, np.exp(1j*np.pi/4)]], dtype=complex))
    U1 = phase.conjugate() * U1

    # Test
    g1 = np.kron(V1, V2)
    g2 = np.array([[1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0]], dtype=complex)
    theta = 2*gamma - np.pi/2
    Ztheta = np.array([[np.exp(1j*theta/2), 0],
                       [0, np.exp(-1j*theta/2)]], dtype=complex)
    kappa = np.pi/2 - 2*alpha
    Ykappa = np.array([[np.cos(kappa/2), np.sin(kappa/2)],
                       [-np.sin(kappa/2), np.cos(kappa/2)]], dtype=complex)
    g3 = np.kron(Ztheta, Ykappa)
    g4 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype=complex)
    zeta = 2*beta - np.pi/2
    Yzeta = np.array([[np.cos(zeta/2), np.sin(zeta/2)],
                      [-np.sin(zeta/2), np.cos(zeta/2)]], dtype=complex)
    g5 = np.kron(np.identity(2), Yzeta)
    g6 = g2
    g7 = np.kron(U1, U2)

    V = np.dot(g2, g1)
    V = np.dot(g3, V)
    V = np.dot(g4, V)
    V = np.dot(g5, V)
    V = np.dot(g6, V)
    V = np.dot(g7, V)

    if np.linalg.norm(V - U*phase.conjugate()) > 1e-6:
        raise MapperError("compiling.two_qubit_kak: " +
                          "sequence incorrect, unknown error")

    v1_param = euler_angles_1q(V1)
    v2_param = euler_angles_1q(V2)
    u1_param = euler_angles_1q(U1)
    u2_param = euler_angles_1q(U2)

    v1_gate = simplify_U(v1_param[0], v1_param[1], v1_param[2])
    v2_gate = simplify_U(v2_param[0], v2_param[1], v2_param[2])
    u1_gate = simplify_U(u1_param[0], u1_param[1], u1_param[2])
    u2_gate = simplify_U(u2_param[0], u2_param[1], u2_param[2])

    return_circuit = []
    return_circuit.append({
        "name": v1_gate[0],
        "args": [0],
        "params": v1_gate[1]
        })
    return_circuit.append({
        "name": v2_gate[0],
        "args": [1],
        "params": v2_gate[1]
        })
    return_circuit.append({
        "name": "cx",
        "args": [1, 0],
        "params": ()
        })
    gate = simplify_U(0.0, 0.0, -2.0*gamma + np.pi/2.0)
    return_circuit.append({
        "name": gate[0],
        "args": [0],
        "params": gate[1]
        })
    gate = simplify_U(-np.pi/2.0 + 2.0*alpha, 0.0, 0.0)
    return_circuit.append({
        "name": gate[0],
        "args": [1],
        "params": gate[1]
    })
    return_circuit.append({
        "name": "cx",
        "args": [0, 1],
        "params": ()
        })
    gate = simplify_U(-2.0*beta + np.pi/2.0, 0.0, 0.0)
    return_circuit.append({
        "name": gate[0],
        "args": [1],
        "params": gate[1]
    })
    return_circuit.append({
        "name": "cx",
        "args": [1, 0],
        "params": ()
        })
    return_circuit.append({
        "name": u1_gate[0],
        "args": [0],
        "params": u1_gate[1]
        })
    return_circuit.append({
        "name": u2_gate[0],
        "args": [1],
        "params": u2_gate[1]
        })

    # Test gate sequence
    V = np.identity(4)
    cx21 = np.array([[1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0]], dtype=complex)
    cx12 = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)
    for gate in return_circuit:
        if gate["name"] == "cx":
            if gate["args"] == [0, 1]:
                V = np.dot(cx12, V)
            else:
                V = np.dot(cx21, V)
        else:
            if gate["args"] == [0]:
                V = np.dot(np.kron(rz_array(gate["params"][2]),
                                   np.identity(2)), V)
                V = np.dot(np.kron(ry_array(gate["params"][0]),
                                   np.identity(2)), V)
                V = np.dot(np.kron(rz_array(gate["params"][1]),
                                   np.identity(2)), V)
            else:
                V = np.dot(np.kron(np.identity(2),
                                   rz_array(gate["params"][2])), V)
                V = np.dot(np.kron(np.identity(2),
                                   ry_array(gate["params"][0])), V)
                V = np.dot(np.kron(np.identity(2),
                                   rz_array(gate["params"][1])), V)
    # Put V in SU(4) and test up to global phase
    V = np.linalg.det(V)**(-1.0/4.0) * V
    if np.linalg.norm(V - U) > 1e-6 and \
       np.linalg.norm(1j*V - U) > 1e-6 and \
       np.linalg.norm(-1*V - U) > 1e-6 and \
       np.linalg.norm(-1j*V - U) > 1e-6:
        raise MapperError("compiling.two_qubit_kak: " +
                          "sequence incorrect, unknown error")

    return return_circuit
