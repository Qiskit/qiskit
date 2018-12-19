# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Methods to assist with compiling tasks.
"""
import math

import numpy as np
import scipy.linalg as la

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
    phase = la.det(unitary_matrix)**(-1.0/2.0)
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
    if la.norm(V - U) > small:
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


def two_qubit_kak(unitary_matrix, verify_gate_sequence=False):
    """Decompose a two-qubit gate over CNOT + SU(2) using the KAK decomposition.

    Based on MATLAB implementation by David Gosset.

    Computes a sequence of 10 single and two qubit gates, including 3 CNOTs,
    which multiply to U, including global phase. Uses Vatan and Williams
    optimal two-qubit circuit (quant-ph/0308006v3). The decomposition algorithm
    which achieves this is explained well in Drury and Love, 0806.4015.

    Args:
        unitary_matrix (ndarray): 4x4 unitary matrix.
        verify_gate_sequence (bool): Verify the decomposition via circuit
                                     gate sequence, optional.

    Returns:
        list: List of gates corresponding to decomposition.

    Raises:
        MapperError: Error in KAK decomposition.
    """
    if unitary_matrix.shape != (4, 4):
        raise MapperError("two_qubit_kak: Expected 4x4 matrix")
    phase = la.det(unitary_matrix)**(-1.0/4.0)
    # Make it in SU(4), correct phase at the end
    U = phase * unitary_matrix
    # B changes to the Bell basis
    B = (1.0/math.sqrt(2)) * np.array([[1, 1j, 0, 0],
                                       [0, 0, 1j, 1],
                                       [0, 0, 1j, -1],
                                       [1, -1j, 0, 0]], dtype=complex)

    # We also need B.conj().T below
    Bdag = B.conj().T
    # U' = Bdag . U . B
    Uprime = Bdag.dot(U.dot(B))
    # M^2 = trans(U') . U'
    M2 = Uprime.T.dot(Uprime)

    # Diagonalize M2
    # Must use diagonalization routine which finds a real orthogonal matrix P
    # when M2 is real.
    D, P = la.eig(M2)
    D = np.diag(D)
    # If det(P) == -1 then in O(4), apply a swap to make P in SO(4)
    if abs(la.det(P)+1) < 1e-5:
        swap = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
        P = P.dot(swap)
        D = swap.dot(D.dot(swap))

    Q = np.sqrt(D)  # array from elementwise sqrt
    # Want to take square root so that Q has determinant 1
    if abs(la.det(Q)+1) < 1e-5:
        Q[0, 0] = -Q[0, 0]

    # Q^-1*P.T = P' -> QP' = P.T (solve for P' using Ax=b)
    Pprime = la.solve(Q, P.T)
    # K' now just U' * P * P'
    Kprime = Uprime.dot(P.dot(Pprime))

    K1 = B.dot(Kprime.dot(P.dot(Bdag)))
    A = B.dot(Q.dot(Bdag))
    K2 = B.dot(P.T.dot(Bdag))
    # KAK = K1 * A * K2
    KAK = K1.dot(A.dot(K2))

    # Verify decomp matches input unitary.
    if la.norm(KAK - U) > 1e-6:
        raise MapperError("two_qubit_kak: KAK decomposition " +
                          "does not return input unitary.")

    # Compute parameters alpha, beta, gamma so that
    # A = exp(i * (alpha * XX + beta * YY + gamma * ZZ))
    xx = np.array([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0]], dtype=complex)

    yy = np.array([[0, 0, 0, -1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [-1, 0, 0, 0]], dtype=complex)

    zz = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]], dtype=complex)

    A_real_tr = A.real.trace()
    alpha = math.atan2(A.dot(xx).imag.trace(), A_real_tr)
    beta = math.atan2(A.dot(yy).imag.trace(), A_real_tr)
    gamma = math.atan2(A.dot(zz).imag.trace(), A_real_tr)

    # K1 = kron(U1, U2) and K2 = kron(V1, V2)
    # Find the matrices U1, U2, V1, V2

    # Find a block in K1 where U1_ij * [U2] is not zero
    L = K1[0:2, 0:2]
    if la.norm(L) < 1e-9:
        L = K1[0:2, 2:4]
        if la.norm(L) < 1e-9:
            L = K1[2:4, 2:4]
    # Remove the U1_ij prefactor
    Q = L.dot(L.conj().T)
    U2 = L / math.sqrt(Q[0, 0].real)

    # Now grab U1 given we know U2
    R = K1.dot(np.kron(np.identity(2), U2.conj().T))
    U1 = np.zeros((2, 2), dtype=complex)
    U1[0, 0] = R[0, 0]
    U1[0, 1] = R[0, 2]
    U1[1, 0] = R[2, 0]
    U1[1, 1] = R[2, 2]

    # Repeat K1 routine for K2
    L = K2[0:2, 0:2]
    if la.norm(L) < 1e-9:
        L = K2[0:2, 2:4]
        if la.norm(L) < 1e-9:
            L = K2[2:4, 2:4]
    Q = np.dot(L, np.transpose(L.conjugate()))
    V2 = L / np.sqrt(Q[0, 0])
    R = np.dot(K2, np.kron(np.identity(2), np.transpose(V2.conjugate())))

    V1 = np.zeros_like(U1)
    V1[0, 0] = R[0, 0]
    V1[0, 1] = R[0, 2]
    V1[1, 0] = R[2, 0]
    V1[1, 1] = R[2, 2]

    if la.norm(np.kron(U1, U2) - K1) > 1e-4:
        raise MapperError("two_qubit_kak: K1 != U1 x U2")
    if la.norm(np.kron(V1, V2) - K2) > 1e-4:
        raise MapperError("two_qubit_kak: K2 != V1 x V2")

    test = la.expm(1j*(alpha * xx + beta * yy + gamma * zz))
    if la.norm(A - test) > 1e-4:
        raise MapperError("two_qubit_kak: " +
                          "Matrix A does not match xx,yy,zz decomposition.")

    # Circuit that implements K1 * A * K2 (up to phase), using
    # Vatan and Williams Fig. 6 of quant-ph/0308006v3
    # Include prefix and suffix single-qubit gates into U2, V1 respectively.

    V2 = np.array([[np.exp(1j*np.pi/4), 0],
                   [0, np.exp(-1j*np.pi/4)]], dtype=complex).dot(V2)
    U1 = U1.dot(np.array([[np.exp(-1j*np.pi/4), 0],
                          [0, np.exp(1j*np.pi/4)]], dtype=complex))

    # Corrects global phase: exp(ipi/4)*phase'
    U1 = U1.dot(np.array([[np.exp(1j*np.pi/4), 0],
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
    Ykappa = np.array([[math.cos(kappa/2), math.sin(kappa/2)],
                       [-math.sin(kappa/2), math.cos(kappa/2)]], dtype=complex)
    g3 = np.kron(Ztheta, Ykappa)
    g4 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype=complex)

    zeta = 2*beta - np.pi/2
    Yzeta = np.array([[math.cos(zeta/2), math.sin(zeta/2)],
                      [-math.sin(zeta/2), math.cos(zeta/2)]], dtype=complex)
    g5 = np.kron(np.identity(2), Yzeta)
    g6 = g2
    g7 = np.kron(U1, U2)

    V = g2.dot(g1)
    V = g3.dot(V)
    V = g4.dot(V)
    V = g5.dot(V)
    V = g6.dot(V)
    V = g7.dot(V)

    if la.norm(V - U*phase.conjugate()) > 1e-6:
        raise MapperError("two_qubit_kak: " +
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

    if verify_gate_sequence:
        # Test gate sequence
        V = np.identity(4, dtype=complex)
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
        V = la.det(V)**(-1.0/4.0) * V
        if la.norm(V - U) > 1e-6 and \
           la.norm(1j*V - U) > 1e-6 and \
           la.norm(-1*V - U) > 1e-6 and \
           la.norm(-1j*V - U) > 1e-6:
            raise MapperError("two_qubit_kak: Circuit implementation" +
                              "does not match input unitary.")

    return return_circuit
