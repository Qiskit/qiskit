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
from typing import Optional
import logging

import numpy as np
import scipy.linalg as la

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import CXGate, RXGate, RYGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.quantum_info.synthesis.weyl import weyl_coordinates
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer, DEFAULT_ATOL

logger = logging.getLogger(__name__)


def decompose_two_qubit_product_gate(special_unitary_matrix: np.ndarray):
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
    phase = np.angle(detL) / 2

    temp = np.kron(L, R)
    deviation = np.abs(np.abs(temp.conj(temp).T.dot(special_unitary_matrix).trace()) - 4)
    if deviation > 1.E-13:
        raise QiskitError("decompose_two_qubit_product_gate: decomposition failed: "
                          "deviation too large: {}".format(deviation))

    return L, R, phase


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
_id = np.array([[1, 0],
                [0, 1]], dtype=complex)


class TwoQubitWeylDecomposition():
    """Decompose two-qubit unitary U = (K1l⊗K1r).Exp(i a xx + i b yy + i c zz).(K2l⊗K2r) , where U ∈
    U(4), (K1l|K1r|K2l|K2r) ∈ SU(2), and we stay in the "Weyl Chamber" 𝜋/4 ≥ a ≥ b ≥ |c|

    This is an abstract factory class that instantiates itself as specialized subclasses based on
    the fidelity, such that the approximation error from specialization has an average gate fidelity
    at least as high as requested. The specialized subclasses have canonical representations thus
    avoiding problems of numerical stability.

    Passing non-None fidelity to specializations is treated as an assertion, raising QiskitError if
    the specialization is more approximate.
    """

    a: float
    b: float
    c: float
    global_phase: float
    K1l: np.ndarray
    K2l: np.ndarray
    K1r: np.ndarray
    K2r: np.ndarray
    unitary_matrix: np.ndarray  # The unitary that was input
    requested_fidelity: Optional[float]  # None for no automatic specialization
    calculated_fidelity: float  # Fidelity after specialization

    _original_decomposition: Optional["TwoQubitWeylDecomposition"]
    _is_flipped_from_original: bool

    def __init_subclass__(cls, **kwargs):
        """Subclasses should be concrete, not factories.

        Have explicitly-instantiated subclass __new__  call base __new__ with fidelity=None"""
        super().__init_subclass__(**kwargs)
        cls.__new__ = (lambda cls, *a, fidelity=None, **k:
                       TwoQubitWeylDecomposition.__new__(cls, *a, fidelity=None, **k))

    def __new__(cls, unitary_matrix, fidelity=(1.-1.E-9)):
        """Perform the Weyl chamber decomposition, and optionally choose a specialized subclass.

        The flip into the Weyl Chamber is described in B. Kraus and J. I. Cirac, Phys. Rev. A 63,
        062309 (2001).

        FIXME: There's a cleaner-seeming method based on choosing branch cuts carefully, in Andrew
        M. Childs, Henry L. Haselgrove, and Michael A. Nielsen, Phys. Rev. A 68, 052311, but I
        wasn't able to get that to work.

        The overall decomposition scheme is taken from Drury and Love, arXiv:0806.4015 [quant-ph].
        """
        pi = np.pi
        pi2 = np.pi/2
        pi4 = np.pi/4

        # Make U be in SU(4)
        U = unitary_matrix.copy()
        detU = la.det(U)
        U *= detU**(-0.25)
        global_phase = np.angle(detU) / 4

        Up = _Bd.dot(U).dot(_B)
        M2 = Up.T.dot(Up)

        # M2 is a symmetric complex matrix. We need to decompose it as M2 = P D P^T where
        # P ∈ SO(4), D is diagonal with unit-magnitude elements.
        # D, P = la.eig(M2)  # this can fail for certain kinds of degeneracy
        for i in range(100):  # FIXME: this randomized algorithm is horrendous
            state = np.random.default_rng(i)
            M2real = state.normal()*M2.real + state.normal()*M2.imag
            _, P = np.linalg.eigh(M2real)
            D = P.T.dot(M2).dot(P).diagonal()
            if np.allclose(P.dot(np.diag(D)).dot(P.T), M2, rtol=0, atol=1.0E-13):
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

        K1l, K1r, phase_l = decompose_two_qubit_product_gate(K1)
        K2l, K2r, phase_r = decompose_two_qubit_product_gate(K2)
        global_phase += phase_l + phase_r

        K1l = K1l.copy()

        # Flip into Weyl chamber
        if cs[0] > pi2:
            cs[0] -= 3*pi2
            K1l = K1l.dot(_ipy)
            K1r = K1r.dot(_ipy)
            global_phase += pi2
        if cs[1] > pi2:
            cs[1] -= 3*pi2
            K1l = K1l.dot(_ipx)
            K1r = K1r.dot(_ipx)
            global_phase += pi2
        conjs = 0
        if cs[0] > pi4:
            cs[0] = pi2-cs[0]
            K1l = K1l.dot(_ipy)
            K2r = _ipy.dot(K2r)
            conjs += 1
            global_phase -= pi2
        if cs[1] > pi4:
            cs[1] = pi2-cs[1]
            K1l = K1l.dot(_ipx)
            K2r = _ipx.dot(K2r)
            conjs += 1
            global_phase += pi2
            if conjs == 1:
                global_phase -= pi
        if cs[2] > pi2:
            cs[2] -= 3*pi2
            K1l = K1l.dot(_ipz)
            K1r = K1r.dot(_ipz)
            global_phase += pi2
            if conjs == 1:
                global_phase -= pi
        if conjs == 1:
            cs[2] = pi2-cs[2]
            K1l = K1l.dot(_ipz)
            K2r = _ipz.dot(K2r)
            global_phase += pi2
        if cs[2] > pi4:
            cs[2] -= pi2
            K1l = K1l.dot(_ipz)
            K1r = K1r.dot(_ipz)
            global_phase -= pi2

        a, b, c = cs[1], cs[0], cs[2]

        # Save the non-specialized decomposition for later comparison
        od = super().__new__(TwoQubitWeylDecomposition)
        od.a = a
        od.b = b
        od.c = c
        od.K1l = K1l
        od.K1r = K1r
        od.K2l = K2l
        od.K2r = K2r
        od.global_phase = global_phase
        od.unitary_matrix = unitary_matrix
        od.requested_fidelity = fidelity

        def is_close(ap, bp, cp):
            da, db, dc = a-ap, b-bp, c-cp
            tr = 4*complex(np.cos(da)*np.cos(db)*np.cos(dc) + np.sin(da)*np.sin(db)*np.sin(dc))
            fid = trace_to_fid(tr)
            return fid >= fidelity

        if fidelity is None:  # Don't specialize if None
            instance = super().__new__(TwoQubitWeylGeneral
                                       if cls is TwoQubitWeylDecomposition else cls)
        elif is_close(0, 0, 0):
            instance = super().__new__(TwoQubitWeylIdEquiv)
        elif is_close(pi4, pi4, pi4) or is_close(pi4, pi4, -pi4):
            instance = super().__new__(TwoQubitWeylSWAPEquiv)
        elif (lambda x: is_close(x, x, x))(_closest_partial_swap(a, b, c)):
            instance = super().__new__(TwoQubitWeylPartialSWAPEquiv)
        elif (lambda x: is_close(x, x, -x))(_closest_partial_swap(a, b, -c)):
            instance = super().__new__(TwoQubitWeylPartialSWAPFlipEquiv)
        elif is_close(a, 0, 0):
            instance = super().__new__(TwoQubitWeylControlledEquiv)
        elif is_close(pi4, pi4, c):
            instance = super().__new__(TwoQubitWeylMirrorControlledEquiv)
        elif is_close((a+b)/2, (a+b)/2, c):
            instance = super().__new__(TwoQubitWeylfSimaabEquiv)
        elif is_close(a, (b+c)/2, (b+c)/2):
            instance = super().__new__(TwoQubitWeylfSimabbEquiv)
        elif is_close(a, (b-c)/2, (c-b)/2):
            instance = super().__new__(TwoQubitWeylfSimabmbEquiv)
        else:
            instance = super().__new__(TwoQubitWeylGeneral)

        instance._original_decomposition = od
        return instance

    def __init__(self, unitary_matrix: np.ndarray, *args, fidelity=None, **kwargs):
        super().__init__(*args, **kwargs)
        od = self._original_decomposition
        self.a, self.b, self.c = od.a, od.b, od.c
        self.K1l, self.K1r = od.K1l, od.K1r
        self.K2l, self.K2r = od.K2l, od.K2r
        self.global_phase = od.global_phase
        self.unitary_matrix = unitary_matrix
        self.requested_fidelity = fidelity
        self._is_flipped_from_original = False
        self.specialize()

        # Update the phase after specialization:
        if self._is_flipped_from_original:
            da, db, dc = (np.pi/2-od.a)-self.a, od.b-self.b, -od.c-self.c
            tr = 4 * complex(np.cos(da)*np.cos(db)*np.cos(dc), np.sin(da)*np.sin(db)*np.sin(dc))
        else:
            da, db, dc = od.a-self.a, od.b-self.b, od.c-self.c
            tr = 4 * complex(np.cos(da)*np.cos(db)*np.cos(dc), np.sin(da)*np.sin(db)*np.sin(dc))
        self.global_phase += np.angle(tr)
        self.calculated_fidelity = trace_to_fid(tr)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Requested fidelity: %s calculated fidelity: %s actual fidelity %s",
                         self.requested_fidelity, self.calculated_fidelity, self.actual_fidelity())
        if self.requested_fidelity and self.calculated_fidelity + 1.E-13 < self.requested_fidelity:
            raise QiskitError(f"{self.__class__.__name__}: "
                              f"calculated fidelity: {self.calculated_fidelity} "
                              f"is worse than requested fidelity: {self.requested_fidelity}.")

    def specialize(self):
        """Make changes to the decomposition to comply with any specialization.

        Do not update the global phase, since gets done in generic __init__()"""
        raise NotImplementedError

    def circuit(self, *, euler_basis='ZYZ', simplify=False, atol=DEFAULT_ATOL, **kwargs):
        """Weyl decomposition in circuit form.

        Arguments are passed to OneQubitEulerDecomposer"""
        oneq_decompose = OneQubitEulerDecomposer(euler_basis)
        circ = QuantumCircuit(2, global_phase=self.global_phase)
        c1l, c1r, c2l, c2r = (oneq_decompose(k, simplify=simplify, atol=atol, **kwargs)
                              for k in (self.K1l, self.K1r, self.K2l, self.K2r))
        circ.compose(c2r, [0], inplace=True)
        circ.compose(c2l, [1], inplace=True)
        if not simplify or abs(self.a) > atol:
            circ.rxx(-self.a*2, 0, 1)
        if not simplify or abs(self.b) > atol:
            circ.ryy(-self.b*2, 0, 1)
        if not simplify or abs(self.c) > atol:
            circ.rzz(-self.c*2, 0, 1)
        circ.compose(c1r, [0], inplace=True)
        circ.compose(c1l, [1], inplace=True)
        return circ

    def actual_fidelity(self, **kwargs):
        """Calculates the actual fidelity of the decomposed circuit to the input unitary"""
        circ = self.circuit(**kwargs)
        trace = np.trace(Operator(circ).data.T.conj() @ self.unitary_matrix)
        if abs(np.imag(trace)) > 1.E-13:
            logger.warning("Trace has a non-zero imaginary component: %s", trace)
        return trace_to_fid(trace)

    def __repr__(self):
        prefix = f"{self.__class__.__name__}("
        prefix_nd = " np.array("
        suffix = "),"
        array = np.array2string(self.unitary_matrix, prefix=prefix+prefix_nd, suffix=suffix,
                                separator=', ', max_line_width=1000,
                                floatmode='maxprec_equal', precision=100)
        return (f"{prefix}{prefix_nd}{array}{suffix}\n"
                f"{' '*len(prefix)}fidelity={self.requested_fidelity})")

    def __str__(self):
        pre = f"{self.__class__.__name__}(\n\t"
        circ_indent = "\n\t".join(self.circuit(simplify=True).draw("text").lines(-1))
        return f"{pre}{circ_indent}\n)"


class TwoQubitWeylIdEquiv(TwoQubitWeylDecomposition):
    """U ~ Ud(0,0,0) ~ Id

    This gate binds 0 parameters, we make it canonical by setting
        K2l = Id ,
        K2r = Id .
    """

    def specialize(self):
        self.a = self.b = self.c = 0.
        self.K1l = self.K1l @ self.K2l
        self.K1r = self.K1r @ self.K2r
        self.K2l = _id.copy()
        self.K2r = _id.copy()


class TwoQubitWeylSWAPEquiv(TwoQubitWeylDecomposition):
    """U ~ Ud(𝜋/4, 𝜋/4, 𝜋/4) ~ U(𝜋/4, 𝜋/4, -𝜋/4) ~ SWAP

    This gate binds 0 parameters, we make it canonical by setting
        K2l = Id ,
        K2r = Id .
    """
    def specialize(self):
        if self.c > 0:
            self.K1l = self.K1l @ self.K2r
            self.K1r = self.K1r @ self.K2l
        else:
            self._is_flipped_from_original = True
            self.K1l = self.K1l @ _ipz @ self.K2r
            self.K1r = self.K1r @ _ipz @ self.K2l
            self.global_phase = self.global_phase + np.pi/2
        self.a = self.b = self.c = np.pi/4
        self.K2l = _id.copy()
        self.K2r = _id.copy()


def _closest_partial_swap(a, b, c):
    """A good approximation to the best value x to get the minimum
    trace distance for Ud(x, x, x) from Ud(a, b, c)
    """
    m = (a + b + c) / 3
    am, bm, cm = a-m, b-m, c-m
    ab, bc, ca = a-b, b-c, c-a

    return m + am * bm * cm * (6 + ab*ab + bc*bc * ca*ca) / 18


class TwoQubitWeylPartialSWAPEquiv(TwoQubitWeylDecomposition):
    """U ~ Ud(α𝜋/4, α𝜋/4, α𝜋/4) ~ SWAP**α

    This gate binds 3 parameters, we make it canonical by setting:
        K2l = Id .
    """
    def specialize(self):
        self.a = self.b = self.c = _closest_partial_swap(self.a, self.b, self.c)
        self.K1l = self.K1l @ self.K2l
        self.K1r = self.K1r @ self.K2l
        self.K2r = self.K2l.T.conj() @ self.K2r
        self.K2l = _id.copy()


class TwoQubitWeylPartialSWAPFlipEquiv(TwoQubitWeylDecomposition):
    """U ~ Ud(α𝜋/4, α𝜋/4, -α𝜋/4) ~ SWAP**α

    (a non-equivalent root of SWAP from the TwoQubitWeylPartialSWAPEquiv
    similar to how x = (±sqrt(x))**2 )

    This gate binds 3 parameters, we make it canonical by setting:
        K2l = Id .
    """

    def specialize(self):
        self.a = self.b = _closest_partial_swap(self.a, self.b, -self.c)
        self.c = -self.a
        self.K1l = self.K1l @ self.K2l
        self.K1r = self.K1r @ _ipz @ self.K2l @ _ipz
        self.K2r = _ipz @ self.K2l.T.conj() @ _ipz @ self.K2r
        self.K2l = _id.copy()


_oneq_xyx = OneQubitEulerDecomposer('XYX')
_oneq_zyz = OneQubitEulerDecomposer('ZYZ')


class TwoQubitWeylControlledEquiv(TwoQubitWeylDecomposition):
    """U ~ Ud(α, 0, 0) ~ Ctrl-U

    This gate binds 4 parameters, we make it canonical by setting:
        K2l = Ry(θl).Rx(λl) ,
        K2r = Ry(θr).Rx(λr) .
    """
    def specialize(self):
        self.b = self.c = 0
        k2ltheta, k2lphi, k2llambda = _oneq_xyx.angles(self.K2l)
        k2rtheta, k2rphi, k2rlambda = _oneq_xyx.angles(self.K2r)
        self.K1l = self.K1l @ np.asarray(RXGate(k2lphi))
        self.K1r = self.K1r @ np.asarray(RXGate(k2rphi))
        self.K2l = np.asarray(RYGate(k2ltheta)) @ np.asarray(RXGate(k2llambda))
        self.K2r = np.asarray(RYGate(k2rtheta)) @ np.asarray(RXGate(k2rlambda))

    def circuit(self, *, euler_basis='XYX', **kwargs):
        return super().circuit(euler_basis=euler_basis, **kwargs)


class TwoQubitWeylMirrorControlledEquiv(TwoQubitWeylDecomposition):
    """U ~ Ud(𝜋/4, 𝜋/4, α) ~ SWAP . Ctrl-U

    This gate binds 4 parameters, we make it canonical by setting:
        K2l = Ry(θl).Rz(λl) ,
        K2r = Ry(θr).Rz(λr) .
    """
    def specialize(self):
        self.a = self.b = np.pi/4
        k2ltheta, k2lphi, k2llambda = _oneq_zyz.angles(self.K2l)
        k2rtheta, k2rphi, k2rlambda = _oneq_zyz.angles(self.K2r)
        self.K1r = self.K1r @ np.asarray(RZGate(k2lphi))
        self.K1l = self.K1l @ np.asarray(RZGate(k2rphi))
        self.K2l = np.asarray(RYGate(k2ltheta)) @ np.asarray(RZGate(k2llambda))
        self.K2r = np.asarray(RYGate(k2rtheta)) @ np.asarray(RZGate(k2rlambda))


# These next 3 gates use the definition of fSim from https://arxiv.org/pdf/2001.08343.pdf eq (1)
class TwoQubitWeylfSimaabEquiv(TwoQubitWeylDecomposition):
    """U ~ Ud(α, α, β), α ≥ |β|

    This gate binds 5 parameters, we make it canonical by setting:
        K2l = Ry(θl).Rz(λl) .
    """
    def specialize(self):
        self.a = self.b = (self.a + self.b)/2
        k2ltheta, k2lphi, k2llambda = _oneq_zyz.angles(self.K2l)
        self.K1r = self.K1r @ np.asarray(RZGate(k2lphi))
        self.K1l = self.K1l @ np.asarray(RZGate(k2lphi))
        self.K2l = np.asarray(RYGate(k2ltheta)) @ np.asarray(RZGate(k2llambda))
        self.K2r = np.asarray(RZGate(-k2lphi)) @ self.K2r


class TwoQubitWeylfSimabbEquiv(TwoQubitWeylDecomposition):
    """U ~ Ud(α, β, β), α ≥ β

    This gate binds 5 parameters, we make it canonical by setting:
        K2l = Ry(θl).Rx(λl) .
    """
    def specialize(self):
        self.b = self.c = (self.b + self.c)/2
        k2ltheta, k2lphi, k2llambda = _oneq_xyx.angles(self.K2l)
        self.K1r = self.K1r @ np.asarray(RXGate(k2lphi))
        self.K1l = self.K1l @ np.asarray(RXGate(k2lphi))
        self.K2l = np.asarray(RYGate(k2ltheta)) @ np.asarray(RXGate(k2llambda))
        self.K2r = np.asarray(RXGate(-k2lphi)) @ self.K2r

    def circuit(self, *, euler_basis='XYX', **kwargs):
        return super().circuit(euler_basis=euler_basis, **kwargs)


class TwoQubitWeylfSimabmbEquiv(TwoQubitWeylDecomposition):
    """U ~ Ud(α, β, -β), α ≥ β ≥ 0

    This gate binds 5 parameters, we make it canonical by setting:
        K2l = Ry(θl).Rx(λl) .
    """
    def specialize(self):
        self.b = (self.b - self.c)/2
        self.c = -self.b
        k2ltheta, k2lphi, k2llambda = _oneq_xyx.angles(self.K2l)
        self.K1r = self.K1r @ _ipz @ np.asarray(RXGate(k2lphi)) @ _ipz
        self.K1l = self.K1l @ np.asarray(RXGate(k2lphi))
        self.K2l = np.asarray(RYGate(k2ltheta)) @ np.asarray(RXGate(k2llambda))
        self.K2r = _ipz @ np.asarray(RXGate(-k2lphi)) @ _ipz @ self.K2r

    def circuit(self, *, euler_basis='XYX', **kwargs):
        return super().circuit(euler_basis=euler_basis, **kwargs)


class TwoQubitWeylGeneral(TwoQubitWeylDecomposition):
    """U has no special symmetry.

    This gate binds all 6 possible parameters, so there is no need to make the single-qubit
    pre-/post-gates canonical.
    """
    def specialize(self):
        pass  # Nothing to do


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
            Valid options are ['ZYZ', 'ZXZ', 'XYX', 'U', 'U3', 'U1X', 'PSX', 'ZSX', 'RR'].
            Default 'U3'.
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
                          "This gate is ~Ud({}, {}, {})".format(basis.a, basis.b, basis.c),
                          stacklevel=2)
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
        return U0r, U0l

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
        return_circuit.global_phase = target_decomposed.global_phase
        return_circuit.global_phase -= best_nbasis * self.basis.global_phase
        if best_nbasis == 2:
            return_circuit.global_phase += np.pi
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
