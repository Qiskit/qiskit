# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Expand 2-qubit Unitary operators into an equivalent
decomposition over SU(2)+fixed 2q basis gate, using the KAK method.

May be exact or approximate expansion. In either case uses the minimal
number of basis applications.

Method is described in Appendix B of Cross, A. W., Bishop, L. S., Sheldon, S., Nation, P. D. &
Gambetta, J. M. Validating quantum computers using randomized model circuits.
arXiv:1811.12926 [quant-ph] (2018).
"""
import math
import warnings

import numpy as np
import scipy.linalg as la

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.quantum_info.synthesis.weyl import weyl_coordinates
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer

_CUTOFF_PRECISION = 1e-12


def euler_angles_1q(unitary_matrix):
    """DEPRECATED: Compute Euler angles for a single-qubit gate.

    Find angles (theta, phi, lambda) such that
    unitary_matrix = phase * Rz(phi) * Ry(theta) * Rz(lambda)

    Args:
        unitary_matrix (ndarray): 2x2 unitary matrix

    Returns:
        tuple: (theta, phi, lambda) Euler angles of SU(2)

    Raises:
        QiskitError: if unitary_matrix not 2x2, or failure
    """
    warnings.warn("euler_angles_1q` is deprecated. "
                  "Use `synthesis.OneQubitEulerDecomposer().angles instead.",
                  DeprecationWarning)
    if unitary_matrix.shape != (2, 2):
        raise QiskitError("euler_angles_1q: expected 2x2 matrix")
    phase = la.det(unitary_matrix)**(-1.0/2.0)
    U = phase * unitary_matrix  # U in SU(2)
    # OpenQASM SU(2) parameterization:
    # U[0, 0] = exp(-i(phi+lambda)/2) * cos(theta/2)
    # U[0, 1] = -exp(-i(phi-lambda)/2) * sin(theta/2)
    # U[1, 0] = exp(i(phi-lambda)/2) * sin(theta/2)
    # U[1, 1] = exp(i(phi+lambda)/2) * cos(theta/2)
    theta = 2 * math.atan2(abs(U[1, 0]), abs(U[0, 0]))

    # Find phi and lambda
    phiplambda = 2 * np.angle(U[1, 1])
    phimlambda = 2 * np.angle(U[1, 0])
    phi = (phiplambda + phimlambda) / 2.0
    lamb = (phiplambda - phimlambda) / 2.0

    # Check the solution
    Rzphi = np.array([[np.exp(-1j*phi/2.0), 0],
                      [0, np.exp(1j*phi/2.0)]], dtype=complex)
    Rytheta = np.array([[np.cos(theta/2.0), -np.sin(theta/2.0)],
                        [np.sin(theta/2.0), np.cos(theta/2.0)]], dtype=complex)
    Rzlambda = np.array([[np.exp(-1j*lamb/2.0), 0],
                         [0, np.exp(1j*lamb/2.0)]], dtype=complex)
    V = np.dot(Rzphi, np.dot(Rytheta, Rzlambda))
    if la.norm(V - U) > _CUTOFF_PRECISION:
        raise QiskitError("compiling.euler_angles_1q incorrect result norm(V-U)={}".
                          format(la.norm(V-U)))
    return theta, phi, lamb


def decompose_two_qubit_product_gate(special_unitary_matrix):
    """Decompose U = Ul⊗Ur where U in SU(4), and Ul, Ur in SU(2).
    Throws QiskitError if this isn't possible.
    """
    # extract the right component
    R = special_unitary_matrix[:2, :2].copy()
    detR = R[0, 0]*R[1, 1] - R[0, 1]*R[1, 0]
    if abs(detR) < 0.1:
        R = special_unitary_matrix[2:, :2].copy()
        detR = R[0, 0]*R[1, 1] - R[0, 1]*R[1, 0]
    if abs(detR) < 0.1:
        raise QiskitError("decompose_two_qubit_product_gate: unable to decompose: detR < 0.1")
    R /= np.sqrt(detR)

    # extract the left component
    temp = np.kron(np.eye(2), R.T.conj())
    temp = special_unitary_matrix.dot(temp)
    L = temp[::2, ::2]
    detL = L[0, 0]*L[1, 1] - L[0, 1]*L[1, 0]
    if abs(detL) < 0.9:
        raise QiskitError("decompose_two_qubit_product_gate: unable to decompose: detL < 0.9")
    L /= np.sqrt(detL)

    temp = np.kron(L, R)
    deviation = np.abs(np.abs(temp.conj(temp).T.dot(special_unitary_matrix).trace()) - 4)
    if deviation > 1.E-13:
        raise QiskitError("decompose_two_qubit_product_gate: decomposition failed: "
                          "deviation too large: {}".format(deviation))

    return L, R


_B = (1.0/math.sqrt(2)) * np.array([[1, 1j, 0, 0],
                                    [0, 0, 1j, 1],
                                    [0, 0, 1j, -1],
                                    [1, -1j, 0, 0]], dtype=complex)
_Bd = _B.T.conj()
_ipx = np.array([[0, 1j],
                 [1j, 0]], dtype=complex)
_ipy = np.array([[0, 1],
                 [-1, 0]], dtype=complex)
_ipz = np.array([[1j, 0],
                 [0, -1j]], dtype=complex)


class TwoQubitWeylDecomposition:
    """ Decompose two-qubit unitary U = (K1l⊗K1r).Exp(i a xx + i b yy + i c zz).(K2l⊗K2r) ,
    where U ∈ U(4), (K1l|K1r|K2l|K2r) ∈ SU(2), and we stay in the "Weyl Chamber"
    𝜋/4 ≥ a ≥ b ≥ |c|
    """

    def __init__(self, unitary_matrix):
        """The flip into the Weyl Chamber is described in B. Kraus and J. I. Cirac,
        Phys. Rev. A 63, 062309 (2001).

        FIXME: There's a cleaner-seeming method based on choosing branch cuts carefully, in
        Andrew M. Childs, Henry L. Haselgrove, and Michael A. Nielsen, Phys. Rev. A 68, 052311,
        but I wasn't able to get that to work.

        The overall decomposition scheme is taken from Drury and Love, arXiv:0806.4015 [quant-ph].
        """
        pi2 = np.pi/2
        pi4 = np.pi/4

        # Make U be in SU(4)
        U = unitary_matrix.copy()
        U *= la.det(U)**(-0.25)

        Up = _Bd.dot(U).dot(_B)
        M2 = Up.T.dot(Up)

        # M2 is a symmetric complex matrix. We need to decompose it as M2 = P D P^T where
        # P ∈ SO(4), D is diagonal with unit-magnitude elements.
        # D, P = la.eig(M2)  # this can fail for certain kinds of degeneracy
        for i in range(100):  # FIXME: this randomized algorithm is horrendous
            state = np.random.default_rng(i)
            M2real = state.normal()*M2.real + state.normal()*M2.imag
            _, P = la.eigh(M2real)
            D = P.T.dot(M2).dot(P).diagonal()
            if np.allclose(P.dot(np.diag(D)).dot(P.T), M2, rtol=1.0e-13, atol=1.0e-13):
                break
        else:
            raise QiskitError("TwoQubitWeylDecomposition: failed to diagonalize M2")

        d = -np.angle(D)/2
        d[3] = -d[0]-d[1]-d[2]
        cs = np.mod((d[:3]+d[3])/2, 2*np.pi)

        # Reorder the eigenvalues to get in the Weyl chamber
        cstemp = np.mod(cs, pi2)
        np.minimum(cstemp, pi2-cstemp, cstemp)
        order = np.argsort(cstemp)[[1, 2, 0]]
        cs = cs[order]
        d[:3] = d[order]
        P[:, :3] = P[:, order]

        # Fix the sign of P to be in SO(4)
        if np.real(la.det(P)) < 0:
            P[:, -1] = -P[:, -1]

        # Find K1, K2 so that U = K1.A.K2, with K being product of single-qubit unitaries
        K1 = _B.dot(Up).dot(P).dot(np.diag(np.exp(1j*d))).dot(_Bd)
        K2 = _B.dot(P.T).dot(_Bd)

        K1l, K1r = decompose_two_qubit_product_gate(K1)
        K2l, K2r = decompose_two_qubit_product_gate(K2)

        K1l = K1l.copy()

        # Flip into Weyl chamber
        if cs[0] > pi2:
            cs[0] -= 3*pi2
            K1l = K1l.dot(_ipy)
            K1r = K1r.dot(_ipy)
        if cs[1] > pi2:
            cs[1] -= 3*pi2
            K1l = K1l.dot(_ipx)
            K1r = K1r.dot(_ipx)
        conjs = 0
        if cs[0] > pi4:
            cs[0] = pi2-cs[0]
            K1l = K1l.dot(_ipy)
            K2r = _ipy.dot(K2r)
            conjs += 1
        if cs[1] > pi4:
            cs[1] = pi2-cs[1]
            K1l = K1l.dot(_ipx)
            K2r = _ipx.dot(K2r)
            conjs += 1
        if cs[2] > pi2:
            cs[2] -= 3*pi2
            K1l = K1l.dot(_ipz)
            K1r = K1r.dot(_ipz)
        if conjs == 1:
            cs[2] = pi2-cs[2]
            K1l = K1l.dot(_ipz)
            K2r = _ipz.dot(K2r)
        if cs[2] > pi4:
            cs[2] -= pi2
            K1l = K1l.dot(_ipz)
            K1r = K1r.dot(_ipz)
        self.a = cs[1]
        self.b = cs[0]
        self.c = cs[2]
        self.K1l = K1l
        self.K1r = K1r
        self.K2l = K2l
        self.K2r = K2r

    def __repr__(self):
        # FIXME: this is worth making prettier since it's very useful for debugging
        return ("{}\n{}\nUd({}, {}, {})\n{}\n{}\n".format(
            np.array_str(self.K1l),
            np.array_str(self.K1r),
            self.a, self.b, self.c,
            np.array_str(self.K2l),
            np.array_str(self.K2r)))


def Ud(a, b, c):
    """Generates the array Exp(i(a xx + b yy + c zz))
    """
    return np.array([[np.exp(1j*c)*np.cos(a-b), 0, 0, 1j*np.exp(1j*c)*np.sin(a-b)],
                     [0, np.exp(-1j*c)*np.cos(a+b), 1j*np.exp(-1j*c)*np.sin(a+b), 0],
                     [0, 1j*np.exp(-1j*c)*np.sin(a+b), np.exp(-1j*c)*np.cos(a+b), 0],
                     [1j*np.exp(1j*c)*np.sin(a-b), 0, 0, np.exp(1j*c)*np.cos(a-b)]], dtype=complex)


def trace_to_fid(trace):
    """Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`
    M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)"""
    return (4 + np.abs(trace)**2)/20


def rz_array(theta):
    """Return numpy array for Rz(theta).

    Rz(theta) = diag(exp(-i*theta/2),exp(i*theta/2))
    """
    return np.array([[np.exp(-1j*theta/2.0), 0],
                     [0, np.exp(1j*theta/2.0)]], dtype=complex)


class TwoQubitBasisDecomposer():
    """A class for decomposing 2-qubit unitaries into minimal number of uses of a 2-qubit
    basis gate.

    Args:
        gate (Gate): Two-qubit gate to be used in the KAK decomposition.
        basis_fidelity (float): Fidelity to be assumed for applications of KAK Gate. Default 1.0.
        euler_basis (str): Basis string to be provided to OneQubitEulerDecomposer for 1Q synthesis.
            Valid options are ['ZYZ', 'ZXZ', 'XYX', 'U3', 'U1X', 'RR']. Default 'U3'.
    """

    def __init__(self, gate, basis_fidelity=1.0, euler_basis=None):
        self.gate = gate
        self.basis_fidelity = basis_fidelity

        basis = self.basis = TwoQubitWeylDecomposition(Operator(gate).data)
        if euler_basis is not None:
            self._decomposer1q = OneQubitEulerDecomposer(euler_basis)
        else:
            self._decomposer1q = OneQubitEulerDecomposer('U3')

        # FIXME: find good tolerances
        self.is_supercontrolled = np.isclose(basis.a, np.pi/4) and np.isclose(basis.c, 0.)

        # Create some useful matrices U1, U2, U3 are equivalent to the basis,
        # expand as Ui = Ki1.Ubasis.Ki2
        b = basis.b
        K11l = 1/(1+1j) * np.array([[-1j*np.exp(-1j*b), np.exp(-1j*b)],
                                    [-1j*np.exp(1j*b), -np.exp(1j*b)]], dtype=complex)
        K11r = 1/np.sqrt(2) * np.array([[1j*np.exp(-1j*b), -np.exp(-1j*b)],
                                        [np.exp(1j*b), -1j*np.exp(1j*b)]], dtype=complex)
        K12l = 1/(1+1j) * np.array([[1j, 1j],
                                    [-1, 1]], dtype=complex)
        K12r = 1/np.sqrt(2) * np.array([[1j, 1],
                                        [-1, -1j]], dtype=complex)
        K32lK21l = 1/np.sqrt(2) * np.array([[1+1j*np.cos(2*b), 1j*np.sin(2*b)],
                                            [1j*np.sin(2*b), 1-1j*np.cos(2*b)]], dtype=complex)
        K21r = 1/(1-1j) * np.array([[-1j*np.exp(-2j*b), np.exp(-2j*b)],
                                    [1j*np.exp(2j*b), np.exp(2j*b)]], dtype=complex)
        K22l = 1/np.sqrt(2) * np.array([[1, -1],
                                        [1, 1]], dtype=complex)
        K22r = np.array([[0, 1], [-1, 0]], dtype=complex)
        K31l = 1/np.sqrt(2) * np.array([[np.exp(-1j*b), np.exp(-1j*b)],
                                        [-np.exp(1j*b), np.exp(1j*b)]], dtype=complex)
        K31r = 1j * np.array([[np.exp(1j*b), 0],
                              [0, -np.exp(-1j*b)]], dtype=complex)
        K32r = 1/(1-1j) * np.array([[np.exp(1j*b), -np.exp(-1j*b)],
                                    [-1j*np.exp(1j*b), -1j*np.exp(-1j*b)]], dtype=complex)
        k1ld = basis.K1l.T.conj()
        k1rd = basis.K1r.T.conj()
        k2ld = basis.K2l.T.conj()
        k2rd = basis.K2r.T.conj()

        # Pre-build the fixed parts of the matrices used in 3-part decomposition
        self.u0l = K31l.dot(k1ld)
        self.u0r = K31r.dot(k1rd)
        self.u1l = k2ld.dot(K32lK21l).dot(k1ld)
        self.u1ra = k2rd.dot(K32r)
        self.u1rb = K21r.dot(k1rd)
        self.u2la = k2ld.dot(K22l)
        self.u2lb = K11l.dot(k1ld)
        self.u2ra = k2rd.dot(K22r)
        self.u2rb = K11r.dot(k1rd)
        self.u3l = k2ld.dot(K12l)
        self.u3r = k2rd.dot(K12r)

        # Pre-build the fixed parts of the matrices used in the 2-part decomposition
        self.q0l = K12l.T.conj().dot(k1ld)
        self.q0r = K12r.T.conj().dot(_ipz).dot(k1rd)
        self.q1la = k2ld.dot(K11l.T.conj())
        self.q1lb = K11l.dot(k1ld)
        self.q1ra = k2rd.dot(_ipz).dot(K11r.T.conj())
        self.q1rb = K11r.dot(k1rd)
        self.q2l = k2ld.dot(K12l)
        self.q2r = k2rd.dot(K12r)

        # Decomposition into different number of gates
        # In the future could use different decomposition functions for different basis classes, etc
        if not self.is_supercontrolled:
            warnings.warn("Only know how to decompose properly for supercontrolled basis gate. "
                          "This gate is ~Ud({}, {}, {})".format(basis.a, basis.b, basis.c))
        self.decomposition_fns = [self.decomp0,
                                  self.decomp1,
                                  self.decomp2_supercontrolled,
                                  self.decomp3_supercontrolled]

    def traces(self, target):
        """Give the expected traces :math:`|Tr(U \\cdot Utarget^dag)|` for different number of
        basis gates."""
        # Future gotcha: extending this to non-supercontrolled basis.
        # Careful: closest distance between a1,b1,c1 and a2,b2,c2 may be between reflections.
        # This doesn't come up if either c1==0 or c2==0 but otherwise be careful.

        return [4*(np.cos(target.a)*np.cos(target.b)*np.cos(target.c) +
                   1j*np.sin(target.a)*np.sin(target.b)*np.sin(target.c)),
                4*(np.cos(np.pi/4-target.a)*np.cos(self.basis.b-target.b)*np.cos(target.c) +
                   1j*np.sin(np.pi/4-target.a)*np.sin(self.basis.b-target.b)*np.sin(target.c)),
                4*np.cos(target.c),
                4]

    @staticmethod
    def decomp0(target):
        """Decompose target ~Ud(x, y, z) with 0 uses of the basis gate.
        Result Ur has trace:
        :math:`|Tr(Ur.Utarget^dag)| = 4|(cos(x)cos(y)cos(z)+ j sin(x)sin(y)sin(z)|`,
        which is optimal for all targets and bases"""

        U0l = target.K1l.dot(target.K2l)
        U0r = target.K1r.dot(target.K2r)

        return np.around(U0r, 13), np.around(U0l, 13)

    def decomp1(self, target):
        """Decompose target ~Ud(x, y, z) with 1 uses of the basis gate ~Ud(a, b, c).
        Result Ur has trace:
        .. math::

            |Tr(Ur.Utarget^dag)| = 4|cos(x-a)cos(y-b)cos(z-c) + j sin(x-a)sin(y-b)sin(z-c)|

        which is optimal for all targets and bases with z==0 or c==0"""
        # FIXME: fix for z!=0 and c!=0 using closest reflection (not always in the Weyl chamber)
        U0l = target.K1l.dot(self.basis.K1l.T.conj())
        U0r = target.K1r.dot(self.basis.K1r.T.conj())
        U1l = self.basis.K2l.T.conj().dot(target.K2l)
        U1r = self.basis.K2r.T.conj().dot(target.K2r)

        return U1r, U1l, U0r, U0l

    def decomp2_supercontrolled(self, target):
        """Decompose target ~Ud(x, y, z) with 2 uses of the basis gate.

        For supercontrolled basis ~Ud(pi/4, b, 0), all b, result Ur has trace
        .. math::

            |Tr(Ur.Utarget^dag)| = 4cos(z)

        which is the optimal approximation for basis of CNOT-class ``~Ud(pi/4, 0, 0)``
        or DCNOT-class ``~Ud(pi/4, pi/4, 0)`` and any target.
        May be sub-optimal for b!=0 (e.g. there exists exact decomposition for any target using B
        ``B~Ud(pi/4, pi/8, 0)``, but not this decomposition.)
        This is an exact decomposition for supercontrolled basis and target ``~Ud(x, y, 0)``.
        No guarantees for non-supercontrolled basis.
        """

        U0l = target.K1l.dot(self.q0l)
        U0r = target.K1r.dot(self.q0r)
        U1l = self.q1la.dot(rz_array(-2*target.a)).dot(self.q1lb)
        U1r = self.q1ra.dot(rz_array(2*target.b)).dot(self.q1rb)
        U2l = self.q2l.dot(target.K2l)
        U2r = self.q2r.dot(target.K2r)

        return U2r, U2l, U1r, U1l, U0r, U0l

    def decomp3_supercontrolled(self, target):
        """Decompose target with 3 uses of the basis.
        This is an exact decomposition for supercontrolled basis ~Ud(pi/4, b, 0), all b,
        and any target. No guarantees for non-supercontrolled basis."""

        U0l = target.K1l.dot(self.u0l)
        U0r = target.K1r.dot(self.u0r)
        U1l = self.u1l
        U1r = self.u1ra.dot(rz_array(-2*target.c)).dot(self.u1rb)
        U2l = self.u2la.dot(rz_array(-2*target.a)).dot(self.u2lb)
        U2r = self.u2ra.dot(rz_array(2*target.b)).dot(self.u2rb)
        U3l = self.u3l.dot(target.K2l)
        U3r = self.u3r.dot(target.K2r)

        return U3r, U3l, U2r, U2l, U1r, U1l, U0r, U0l

    def __call__(self, target, basis_fidelity=None):
        """Decompose a two-qubit unitary over fixed basis + SU(2) using the best approximation given
        that each basis application has a finite fidelity.
        """
        basis_fidelity = basis_fidelity or self.basis_fidelity
        if hasattr(target, 'to_operator'):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            target = target.to_operator().data
        if hasattr(target, 'to_matrix'):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            target = target.to_matrix()
        # Convert to numpy array incase not already an array
        target = np.asarray(target, dtype=complex)
        # Check input is a 2-qubit unitary
        if target.shape != (4, 4):
            raise QiskitError("TwoQubitBasisDecomposer: expected 4x4 matrix for target")
        if not is_unitary_matrix(target):
            raise QiskitError("TwoQubitBasisDecomposer: target matrix is not unitary.")

        target_decomposed = TwoQubitWeylDecomposition(target)
        traces = self.traces(target_decomposed)
        expected_fidelities = [trace_to_fid(traces[i]) * basis_fidelity**i for i in range(4)]

        best_nbasis = np.argmax(expected_fidelities)
        decomposition = self.decomposition_fns[best_nbasis](target_decomposed)
        decomposition_euler = [self._decomposer1q(x) for x in decomposition]

        q = QuantumRegister(2)
        return_circuit = QuantumCircuit(q)
        for i in range(best_nbasis):
            return_circuit.compose(decomposition_euler[2*i], [q[0]], inplace=True)
            return_circuit.compose(decomposition_euler[2*i+1], [q[1]], inplace=True)
            return_circuit.append(self.gate, [q[0], q[1]])
        return_circuit.compose(decomposition_euler[2*best_nbasis], [q[0]], inplace=True)
        return_circuit.compose(decomposition_euler[2*best_nbasis+1], [q[1]], inplace=True)

        return return_circuit

    def num_basis_gates(self, unitary):
        """ Computes the number of basis gates needed in
        a decomposition of input unitary
        """
        if hasattr(unitary, 'to_operator'):
            unitary = unitary.to_operator().data
        if hasattr(unitary, 'to_matrix'):
            unitary = unitary.to_matrix()
        unitary = np.asarray(unitary, dtype=complex)
        a, b, c = weyl_coordinates(unitary)[:]
        traces = [4*(np.cos(a)*np.cos(b)*np.cos(c)+1j*np.sin(a)*np.sin(b)*np.sin(c)),
                  4*(np.cos(np.pi/4-a)*np.cos(self.basis.b-b)*np.cos(c) +
                     1j*np.sin(np.pi/4-a)*np.sin(self.basis.b-b)*np.sin(c)),
                  4*np.cos(c),
                  4]
        return np.argmax([trace_to_fid(traces[i]) * self.basis_fidelity**i for i in range(4)])


two_qubit_cnot_decompose = TwoQubitBasisDecomposer(CXGate())
