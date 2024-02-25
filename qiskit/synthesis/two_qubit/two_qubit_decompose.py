# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
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
from __future__ import annotations
import cmath
import math
import io
import base64
import warnings
from typing import ClassVar, Optional, Type

import logging

import numpy as np

from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.circuit.library.standard_gates import CXGate, RXGate, RYGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.two_qubit.weyl import transform_to_magic_basis
from qiskit.synthesis.one_qubit.one_qubit_decompose import (
    OneQubitEulerDecomposer,
    DEFAULT_ATOL,
)
from qiskit._accelerate import two_qubit_decompose

logger = logging.getLogger(__name__)


def decompose_two_qubit_product_gate(special_unitary_matrix: np.ndarray):
    r"""Decompose :math:`U = U_l \otimes U_r` where :math:`U \in SU(4)`,
    and :math:`U_l,~U_r \in SU(2)`.

    Args:
        special_unitary_matrix: special unitary matrix to decompose
    Raises:
        QiskitError: if decomposition isn't possible.
    """
    special_unitary_matrix = np.asarray(special_unitary_matrix, dtype=complex)
    # extract the right component
    R = special_unitary_matrix[:2, :2].copy()
    detR = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(detR) < 0.1:
        R = special_unitary_matrix[2:, :2].copy()
        detR = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(detR) < 0.1:
        raise QiskitError("decompose_two_qubit_product_gate: unable to decompose: detR < 0.1")
    R /= np.sqrt(detR)

    # extract the left component
    temp = np.kron(np.eye(2), R.T.conj())
    temp = special_unitary_matrix.dot(temp)
    L = temp[::2, ::2]
    detL = L[0, 0] * L[1, 1] - L[0, 1] * L[1, 0]
    if abs(detL) < 0.9:
        raise QiskitError("decompose_two_qubit_product_gate: unable to decompose: detL < 0.9")
    L /= np.sqrt(detL)
    phase = cmath.phase(detL) / 2

    temp = np.kron(L, R)
    deviation = abs(abs(temp.conj().T.dot(special_unitary_matrix).trace()) - 4)
    if deviation > 1.0e-13:
        raise QiskitError(
            "decompose_two_qubit_product_gate: decomposition failed: "
            "deviation too large: {}".format(deviation)
        )

    return L, R, phase


_ipx = np.array([[0, 1j], [1j, 0]], dtype=complex)
_ipy = np.array([[0, 1], [-1, 0]], dtype=complex)
_ipz = np.array([[1j, 0], [0, -1j]], dtype=complex)
_id = np.array([[1, 0], [0, 1]], dtype=complex)


class TwoQubitWeylDecomposition(two_qubit_decompose.TwoQubitWeylDecomposition):
    r"""Two-qubit Weyl decomposition.

    Decompose two-qubit unitary

    .. math::

        U = ({K_1}^l \otimes {K_1}^r) e^{(i a XX + i b YY + i c ZZ)} ({K_2}^l \otimes {K_2}^r)

    where

    .. math::

        U \in U(4),~
        {K_1}^l, {K_1}^r, {K_2}^l, {K_2}^r \in SU(2)

    and we stay in the "Weyl Chamber"

    .. math::

        \pi /4 \geq a \geq b \geq |c|

    This is an abstract factory class that instantiates itself as specialized subclasses based on
    the fidelity, such that the approximation error from specialization has an average gate fidelity
    at least as high as requested. The specialized subclasses have unique canonical representations
    thus avoiding problems of numerical stability.

    Passing non-None fidelity to specializations is treated as an assertion, raising QiskitError if
    forcing the specialization is more approximate than asserted.

    References:
        1. Cross, A. W., Bishop, L. S., Sheldon, S., Nation, P. D. & Gambetta, J. M.,
           *Validating quantum computers using randomized model circuits*,
           `arXiv:1811.12926 [quant-ph] <https://arxiv.org/abs/1811.12926>`_
        2. B. Kraus, J. I. Cirac, *Optimal Creation of Entanglement Using a Two-Qubit Gate*,
           `arXiv:0011050 [quant-ph] <https://arxiv.org/abs/quant-ph/0011050>`_
        3. B. Drury, P. J. Love, *Constructive Quantum Shannon Decomposition from Cartan
           Involutions*, `arXiv:0806.4015 [quant-ph] <https://arxiv.org/abs/0806.4015>`_

    """

    # The parameters of the decomposition:
    a: float
    b: float
    c: float
    global_phase: float
    K1l: np.ndarray
    K2l: np.ndarray
    K1r: np.ndarray
    K2r: np.ndarray

    unitary_matrix: np.ndarray  # The unitary that was input
    requested_fidelity: Optional[float]  # None means no automatic specialization
    calculated_fidelity: float  # Fidelity after specialization

    def circuit(
        self, *, euler_basis: str | None = None, simplify: bool = False, atol: float = DEFAULT_ATOL
    ) -> QuantumCircuit:
        """Returns Weyl decomposition in circuit form."""
        circuit_sequence = super().circuit()
        circ = QuantumCircuit(2, global_phase=circuit_sequence.global_phase)
        for name, params, qubits in circuit_sequence:
            if qubits[0] == qubits[1]:
                qargs = (qubits[0],)
            else:
                qargs = tuple(qubits)
            getattr(circ, name)(*params, *qargs)
        return circ

    def actual_fidelity(self, **kwargs) -> float:
        """Calculates the actual fidelity of the decomposed circuit to the input unitary."""
        circ = self.circuit(**kwargs)
        trace = np.trace(Operator(circ).data.T.conj() @ self.unitary_matrix)
        return trace_to_fid(trace)

    def __repr__(self):
        """Represent with enough precision to allow copy-paste debugging of all corner cases"""
        prefix = f"{type(self).__qualname__}.from_bytes("
        with io.BytesIO() as f:
            np.save(f, self.unitary_matrix, allow_pickle=False)
            b64 = base64.encodebytes(f.getvalue()).splitlines()
        b64ascii = [repr(x) for x in b64]
        b64ascii[-1] += ","
        pretty = [f"# {x.rstrip()}" for x in str(self).splitlines()]
        indent = "\n" + " " * 4
        lines = (
            [prefix]
            + pretty
            + b64ascii
            + [
                f"requested_fidelity={self.requested_fidelity},",
                f"calculated_fidelity={self.calculated_fidelity},",
                f"actual_fidelity={self.actual_fidelity()},",
                f"abc={(self.a, self.b, self.c)})",
            ]
        )
        return indent.join(lines)

    @classmethod
    def from_bytes(
        cls, bytes_in: bytes, *, requested_fidelity: float, **kwargs
    ) -> "TwoQubitWeylDecomposition":
        """Decode bytes into :class:`.TwoQubitWeylDecomposition`."""
        # Used by __repr__
        del kwargs  # Unused (just for display)
        b64 = base64.decodebytes(bytes_in)
        with io.BytesIO(b64) as f:
            arr = np.load(f, allow_pickle=False)
        return cls(arr, fidelity=requested_fidelity)

    def __str__(self):
        pre = f"{self.__class__.__name__}(\n\t"
        circ_indent = "\n\t".join(self.circuit(simplify=True).draw("text").lines(-1))
        return f"{pre}{circ_indent}\n)"


class TwoQubitControlledUDecomposer:
    r"""Decompose two-qubit unitary in terms of a desired
    :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}`
    gate that is locally equivalent to an :class:`.RXXGate`."""

    def __init__(self, rxx_equivalent_gate: Type[Gate]):
        r"""Initialize the KAK decomposition.

        Args:
            rxx_equivalent_gate: Gate that is locally equivalent to an :class:`.RXXGate`:
            :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}` gate.
        Raises:
            QiskitError: If the gate is not locally equivalent to an :class:`.RXXGate`.
        """
        atol = DEFAULT_ATOL

        scales, test_angles, scale = [], [0.2, 0.3, np.pi / 2], None

        for test_angle in test_angles:
            # Check that gate takes a single angle parameter
            try:
                rxx_equivalent_gate(test_angle, label="foo")
            except TypeError as _:
                raise QiskitError("Equivalent gate needs to take exactly 1 angle parameter.") from _
            decomp = TwoQubitWeylDecomposition(rxx_equivalent_gate(test_angle).to_matrix())

            circ = QuantumCircuit(2)
            circ.rxx(test_angle, 0, 1)
            decomposer_rxx = TwoQubitWeylDecomposition(Operator(circ).data, fidelity=None, specialization=two_qubit_decompose.Specializations.ControlledEquiv)

            circ = QuantumCircuit(2)
            circ.append(rxx_equivalent_gate(test_angle), qargs=[0, 1])
            decomposer_equiv = TwoQubitWeylDecomposition(Operator(circ).data, fidelity=None, specialization=two_qubit_decompose.Specializations.ControlledEquiv)

            scale = decomposer_rxx.a / decomposer_equiv.a

            if abs(decomp.a * 2 - test_angle / scale) > atol:
                raise QiskitError(
                    f"{rxx_equivalent_gate.__name__} is not equivalent to an RXXGate."
                )

            scales.append(scale)

        # Check that all three tested angles give the same scale
        if not np.allclose(scales, [scale] * len(test_angles)):
            raise QiskitError(
                f"Cannot initialize {self.__class__.__name__}: with gate {rxx_equivalent_gate}. "
                "Inconsistent scaling parameters in checks."
            )

        self.scale = scales[0]

        self.rxx_equivalent_gate = rxx_equivalent_gate

    def __call__(self, unitary, *, atol=DEFAULT_ATOL) -> QuantumCircuit:
        """Returns the Weyl decomposition in circuit form.

        Note: atol ist passed to OneQubitEulerDecomposer.
        """

        # pylint: disable=attribute-defined-outside-init
        self.decomposer = TwoQubitWeylDecomposition(np.asarray(unitary, dtype=complex))

        oneq_decompose = OneQubitEulerDecomposer("ZYZ")
        c1l, c1r, c2l, c2r = (
            oneq_decompose(k, atol=atol)
            for k in (
                self.decomposer.K1l,
                self.decomposer.K1r,
                self.decomposer.K2l,
                self.decomposer.K2r,
            )
        )
        circ = QuantumCircuit(2, global_phase=self.decomposer.global_phase)
        circ.compose(c2r, [0], inplace=True)
        circ.compose(c2l, [1], inplace=True)
        self._weyl_gate(circ)
        circ.compose(c1r, [0], inplace=True)
        circ.compose(c1l, [1], inplace=True)
        return circ

    def _to_rxx_gate(self, angle: float) -> QuantumCircuit:
        """
        Takes an angle and returns the circuit equivalent to an RXXGate with the
        RXX equivalent gate as the two-qubit unitary.

        Args:
            angle: Rotation angle (in this case one of the Weyl parameters a, b, or c)

        Returns:
            Circuit: Circuit equivalent to an RXXGate.

        Raises:
            QiskitError: If the circuit is not equivalent to an RXXGate.
        """

        # The user-provided RXXGate equivalent gate may be locally equivalent to the RXXGate
        # but with some scaling in the rotation angle. For example, RXXGate(angle) has Weyl
        # parameters (angle, 0, 0) for angle in [0, pi/2] but the user provided gate, i.e.
        # :code:`self.rxx_equivalent_gate(angle)` might produce the Weyl parameters
        # (scale * angle, 0, 0) where scale != 1. This is the case for the CPhaseGate.

        circ = QuantumCircuit(2)
        circ.append(self.rxx_equivalent_gate(self.scale * angle), qargs=[0, 1])
        decomposer_inv = TwoQubitWeylDecomposition(Operator(circ).data)

        oneq_decompose = OneQubitEulerDecomposer("ZYZ")

        # Express the RXXGate in terms of the user-provided RXXGate equivalent gate.
        rxx_circ = QuantumCircuit(2, global_phase=-decomposer_inv.global_phase)
        rxx_circ.compose(oneq_decompose(decomposer_inv.K2r).inverse(), inplace=True, qubits=[0])
        rxx_circ.compose(oneq_decompose(decomposer_inv.K2l).inverse(), inplace=True, qubits=[1])
        rxx_circ.compose(circ, inplace=True)
        rxx_circ.compose(oneq_decompose(decomposer_inv.K1r).inverse(), inplace=True, qubits=[0])
        rxx_circ.compose(oneq_decompose(decomposer_inv.K1l).inverse(), inplace=True, qubits=[1])

        return rxx_circ

    def _weyl_gate(self, circ: QuantumCircuit, atol=1.0e-13):
        """Appends U_d(a, b, c) to the circuit."""

        circ_rxx = self._to_rxx_gate(-2 * self.decomposer.a)
        circ.compose(circ_rxx, inplace=True)

        # translate the RYYGate(b) into a circuit based on the desired Ctrl-U gate.
        if abs(self.decomposer.b) > atol:
            circ_ryy = QuantumCircuit(2)
            circ_ryy.sdg(0)
            circ_ryy.sdg(1)
            circ_ryy.compose(self._to_rxx_gate(-2 * self.decomposer.b), inplace=True)
            circ_ryy.s(0)
            circ_ryy.s(1)
            circ.compose(circ_ryy, inplace=True)

        # translate the RZZGate(c) into a circuit based on the desired Ctrl-U gate.
        if abs(self.decomposer.c) > atol:
            # Since the Weyl chamber is here defined as a > b > |c| we may have
            # negative c. This will cause issues in _to_rxx_gate
            # as TwoQubitWeylControlledEquiv will map (c, 0, 0) to (|c|, 0, 0).
            # We therefore produce RZZGate(|c|) and append its inverse to the
            # circuit if c < 0.
            gamma, invert = -2 * self.decomposer.c, False
            if gamma > 0:
                gamma *= -1
                invert = True

            circ_rzz = QuantumCircuit(2)
            circ_rzz.h(0)
            circ_rzz.h(1)
            circ_rzz.compose(self._to_rxx_gate(gamma), inplace=True)
            circ_rzz.h(0)
            circ_rzz.h(1)

            if invert:
                circ.compose(circ_rzz.inverse(), inplace=True)
            else:
                circ.compose(circ_rzz, inplace=True)

        return circ


def Ud(a, b, c):
    r"""Generates the array :math:`e^{(i a XX + i b YY + i c ZZ)}`"""
    return np.array(
        [
            [cmath.exp(1j * c) * math.cos(a - b), 0, 0, 1j * cmath.exp(1j * c) * math.sin(a - b)],
            [0, cmath.exp(-1j * c) * math.cos(a + b), 1j * cmath.exp(-1j * c) * math.sin(a + b), 0],
            [0, 1j * cmath.exp(-1j * c) * math.sin(a + b), cmath.exp(-1j * c) * math.cos(a + b), 0],
            [1j * cmath.exp(1j * c) * math.sin(a - b), 0, 0, cmath.exp(1j * c) * math.cos(a - b)],
        ],
        dtype=complex,
    )


def trace_to_fid(trace):
    r"""Average gate fidelity is

    .. math::

        \bar{F} = \frac{d + |\mathrm{Tr} (U_\text{target} \cdot U^{\dag})|^2}{d(d+1)}

    M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)"""
    return (4 + abs(trace) ** 2) / 20


def rz_array(theta):
    """Return numpy array for Rz(theta).

    Rz(theta) = diag(exp(-i*theta/2),exp(i*theta/2))
    """
    return np.array(
        [[cmath.exp(-1j * theta / 2.0), 0], [0, cmath.exp(1j * theta / 2.0)]], dtype=complex
    )


class TwoQubitBasisDecomposer:
    """A class for decomposing 2-qubit unitaries into minimal number of uses of a 2-qubit
    basis gate.

    Args:
        gate: Two-qubit gate to be used in the KAK decomposition.
        basis_fidelity: Fidelity to be assumed for applications of KAK Gate. Defaults to ``1.0``.
        euler_basis: Basis string to be provided to :class:`.OneQubitEulerDecomposer` for 1Q synthesis.
            Valid options are [``'ZYZ'``, ``'ZXZ'``, ``'XYX'``, ``'U'``, ``'U3'``, ``'U1X'``,
            ``'PSX'``, ``'ZSX'``, ``'RR'``].
        pulse_optimize: If ``True``, try to do decomposition which minimizes
            local unitaries in between entangling gates. This will raise an exception if an
            optimal decomposition is not implemented. Currently, only [{CX, SX, RZ}] is known.
            If ``False``, don't attempt optimization. If ``None``, attempt optimization but don't raise
            if unknown.

    .. automethod:: __call__
    """

    def __init__(
        self,
        gate: Gate,
        basis_fidelity: float = 1.0,
        euler_basis: str = "U",
        pulse_optimize: bool | None = None,
    ):
        self.gate = gate
        self.basis_fidelity = basis_fidelity
        self.pulse_optimize = pulse_optimize

        basis = self.basis = TwoQubitWeylDecomposition(Operator(gate).data)
        self._decomposer1q = OneQubitEulerDecomposer(euler_basis)

        # FIXME: find good tolerances
        self.is_supercontrolled = math.isclose(basis.a, np.pi / 4) and math.isclose(basis.c, 0.0)

        # Create some useful matrices U1, U2, U3 are equivalent to the basis,
        # expand as Ui = Ki1.Ubasis.Ki2
        b = basis.b
        K11l = (
            1
            / (1 + 1j)
            * np.array(
                [
                    [-1j * cmath.exp(-1j * b), cmath.exp(-1j * b)],
                    [-1j * cmath.exp(1j * b), -cmath.exp(1j * b)],
                ],
                dtype=complex,
            )
        )
        K11r = (
            1
            / math.sqrt(2)
            * np.array(
                [
                    [1j * cmath.exp(-1j * b), -cmath.exp(-1j * b)],
                    [cmath.exp(1j * b), -1j * cmath.exp(1j * b)],
                ],
                dtype=complex,
            )
        )
        K12l = 1 / (1 + 1j) * np.array([[1j, 1j], [-1, 1]], dtype=complex)
        K12r = 1 / math.sqrt(2) * np.array([[1j, 1], [-1, -1j]], dtype=complex)
        K32lK21l = (
            1
            / math.sqrt(2)
            * np.array(
                [
                    [1 + 1j * np.cos(2 * b), 1j * np.sin(2 * b)],
                    [1j * np.sin(2 * b), 1 - 1j * np.cos(2 * b)],
                ],
                dtype=complex,
            )
        )
        K21r = (
            1
            / (1 - 1j)
            * np.array(
                [
                    [-1j * cmath.exp(-2j * b), cmath.exp(-2j * b)],
                    [1j * cmath.exp(2j * b), cmath.exp(2j * b)],
                ],
                dtype=complex,
            )
        )
        K22l = 1 / math.sqrt(2) * np.array([[1, -1], [1, 1]], dtype=complex)
        K22r = np.array([[0, 1], [-1, 0]], dtype=complex)
        K31l = (
            1
            / math.sqrt(2)
            * np.array(
                [[cmath.exp(-1j * b), cmath.exp(-1j * b)], [-cmath.exp(1j * b), cmath.exp(1j * b)]],
                dtype=complex,
            )
        )
        K31r = 1j * np.array([[cmath.exp(1j * b), 0], [0, -cmath.exp(-1j * b)]], dtype=complex)
        K32r = (
            1
            / (1 - 1j)
            * np.array(
                [
                    [cmath.exp(1j * b), -cmath.exp(-1j * b)],
                    [-1j * cmath.exp(1j * b), -1j * cmath.exp(-1j * b)],
                ],
                dtype=complex,
            )
        )
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
            warnings.warn(
                "Only know how to decompose properly for supercontrolled basis gate. "
                "This gate is ~Ud({}, {}, {})".format(basis.a, basis.b, basis.c),
                stacklevel=2,
            )
        self.decomposition_fns = [
            self.decomp0,
            self.decomp1,
            self.decomp2_supercontrolled,
            self.decomp3_supercontrolled,
        ]
        self._rqc = None

    def traces(self, target):
        r"""
        Give the expected traces :math:`\Big\vert\text{Tr}(U \cdot U_\text{target}^{\dag})\Big\vert`
        for a different number of basis gates.
        """
        # Future gotcha: extending this to non-supercontrolled basis.
        # Careful: closest distance between a1,b1,c1 and a2,b2,c2 may be between reflections.
        # This doesn't come up if either c1==0 or c2==0 but otherwise be careful.
        ta, tb, tc = target.a, target.b, target.c
        bb = self.basis.b
        return [
            4
            * complex(
                math.cos(ta) * math.cos(tb) * math.cos(tc),
                math.sin(ta) * math.sin(tb) * math.sin(tc),
            ),
            4
            * complex(
                math.cos(math.pi / 4 - ta) * math.cos(bb - tb) * math.cos(tc),
                math.sin(math.pi / 4 - ta) * math.sin(bb - tb) * math.sin(tc),
            ),
            4 * math.cos(tc),
            4,
        ]

    @staticmethod
    def decomp0(target):
        r"""
        Decompose target :math:`\sim U_d(x, y, z)` with :math:`0` uses of the basis gate.
        Result :math:`U_r` has trace:

        .. math::

            \Big\vert\text{Tr}(U_r\cdot U_\text{target}^{\dag})\Big\vert =
            4\Big\vert (\cos(x)\cos(y)\cos(z)+ j \sin(x)\sin(y)\sin(z)\Big\vert

        which is optimal for all targets and bases
        """

        U0l = target.K1l.dot(target.K2l)
        U0r = target.K1r.dot(target.K2r)
        return U0r, U0l

    def decomp1(self, target):
        r"""Decompose target :math:`\sim U_d(x, y, z)` with :math:`1` use of the basis gate
        :math:`\sim U_d(a, b, c)`.
        Result :math:`U_r` has trace:

        .. math::

            \Big\vert\text{Tr}(U_r \cdot U_\text{target}^{\dag})\Big\vert =
            4\Big\vert \cos(x-a)\cos(y-b)\cos(z-c) + j \sin(x-a)\sin(y-b)\sin(z-c)\Big\vert

        which is optimal for all targets and bases with ``z==0`` or ``c==0``.
        """
        # FIXME: fix for z!=0 and c!=0 using closest reflection (not always in the Weyl chamber)
        U0l = target.K1l.dot(self.basis.K1l.T.conj())
        U0r = target.K1r.dot(self.basis.K1r.T.conj())
        U1l = self.basis.K2l.T.conj().dot(target.K2l)
        U1r = self.basis.K2r.T.conj().dot(target.K2r)

        return U1r, U1l, U0r, U0l

    def decomp2_supercontrolled(self, target):
        r"""
        Decompose target :math:`\sim U_d(x, y, z)` with :math:`2` uses of the basis gate.

        For supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b, result :math:`U_r` has trace

        .. math::

            \Big\vert\text{Tr}(U_r \cdot U_\text{target}^\dag) \Big\vert = 4\cos(z)

        which is the optimal approximation for basis of CNOT-class :math:`\sim U_d(\pi/4, 0, 0)`
        or DCNOT-class :math:`\sim U_d(\pi/4, \pi/4, 0)` and any target. It may
        be sub-optimal for :math:`b \neq 0` (i.e. there exists an exact decomposition for any target
        using :math:`B \sim U_d(\pi/4, \pi/8, 0)`, but it may not be this decomposition).
        This is an exact decomposition for supercontrolled basis and target :math:`\sim U_d(x, y, 0)`.
        No guarantees for non-supercontrolled basis.
        """

        U0l = target.K1l.dot(self.q0l)
        U0r = target.K1r.dot(self.q0r)
        U1l = self.q1la.dot(rz_array(-2 * target.a)).dot(self.q1lb)
        U1r = self.q1ra.dot(rz_array(2 * target.b)).dot(self.q1rb)
        U2l = self.q2l.dot(target.K2l)
        U2r = self.q2r.dot(target.K2r)

        return U2r, U2l, U1r, U1l, U0r, U0l

    def decomp3_supercontrolled(self, target):
        r"""
        Decompose target with :math:`3` uses of the basis.
        This is an exact decomposition for supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b,
        and any target. No guarantees for non-supercontrolled basis.
        """

        U0l = target.K1l.dot(self.u0l)
        U0r = target.K1r.dot(self.u0r)
        U1l = self.u1l
        U1r = self.u1ra.dot(rz_array(-2 * target.c)).dot(self.u1rb)
        U2l = self.u2la.dot(rz_array(-2 * target.a)).dot(self.u2lb)
        U2r = self.u2ra.dot(rz_array(2 * target.b)).dot(self.u2rb)
        U3l = self.u3l.dot(target.K2l)
        U3r = self.u3r.dot(target.K2r)

        return U3r, U3l, U2r, U2l, U1r, U1l, U0r, U0l

    def __call__(
        self,
        unitary: Operator | np.ndarray,
        basis_fidelity: float | None = None,
        approximate: bool = True,
        *,
        _num_basis_uses: int | None = None,
    ) -> QuantumCircuit:
        r"""Decompose a two-qubit ``unitary`` over fixed basis and :math:`SU(2)` using the best
        approximation given that each basis application has a finite ``basis_fidelity``.

        Args:
            unitary (Operator or ndarray): :math:`4 \times 4` unitary to synthesize.
            basis_fidelity (float or None): Fidelity to be assumed for applications of KAK Gate.
                If given, overrides ``basis_fidelity`` given at init.
            approximate (bool): Approximates if basis fidelities are less than 1.0.
            _num_basis_uses (int): force a particular approximation by passing a number in [0, 3].

        Returns:
            QuantumCircuit: Synthesized quantum circuit.

        Raises:
            QiskitError: if ``pulse_optimize`` is True but we don't know how to do it.
        """
        basis_fidelity = basis_fidelity or self.basis_fidelity
        if approximate is False:
            basis_fidelity = 1.0
        unitary = np.asarray(unitary, dtype=complex)

        target_decomposed = TwoQubitWeylDecomposition(unitary)
        traces = self.traces(target_decomposed)
        expected_fidelities = [trace_to_fid(traces[i]) * basis_fidelity**i for i in range(4)]

        best_nbasis = int(np.argmax(expected_fidelities))
        if _num_basis_uses is not None:
            best_nbasis = _num_basis_uses
        decomposition = self.decomposition_fns[best_nbasis](target_decomposed)

        # attempt pulse optimal decomposition
        try:
            if self.pulse_optimize in {None, True}:
                return_circuit = self._pulse_optimal_chooser(
                    best_nbasis, decomposition, target_decomposed
                )
                if return_circuit:
                    return return_circuit
        except QiskitError:
            if self.pulse_optimize:
                raise

        # do default decomposition
        q = QuantumRegister(2)
        decomposition_euler = [self._decomposer1q._decompose(x) for x in decomposition]
        return_circuit = QuantumCircuit(q)
        return_circuit.global_phase = target_decomposed.global_phase
        return_circuit.global_phase -= best_nbasis * self.basis.global_phase
        if best_nbasis == 2:
            return_circuit.global_phase += np.pi
        for i in range(best_nbasis):
            return_circuit.compose(decomposition_euler[2 * i], [q[0]], inplace=True)
            return_circuit.compose(decomposition_euler[2 * i + 1], [q[1]], inplace=True)
            return_circuit.append(self.gate, [q[0], q[1]])
        return_circuit.compose(decomposition_euler[2 * best_nbasis], [q[0]], inplace=True)
        return_circuit.compose(decomposition_euler[2 * best_nbasis + 1], [q[1]], inplace=True)
        return return_circuit

    def _pulse_optimal_chooser(
        self, best_nbasis, decomposition, target_decomposed
    ) -> QuantumCircuit:
        """Determine method to find pulse optimal circuit. This method may be
        removed once a more general approach is used.

        Returns:
            QuantumCircuit: pulse optimal quantum circuit.
            None: Probably ``nbasis==1`` and original circuit is fine.

        Raises:
            QiskitError: Decomposition for selected basis not implemented.
        """
        circuit = None
        if self.pulse_optimize and best_nbasis in {0, 1}:
            # already pulse optimal
            return None
        elif self.pulse_optimize and best_nbasis > 3:
            raise QiskitError(
                f"Unexpected number of entangling gates ({best_nbasis}) in decomposition."
            )
        if self._decomposer1q.basis in {"ZSX", "ZSXX"}:
            if isinstance(self.gate, CXGate):
                if best_nbasis == 3:
                    circuit = self._get_sx_vz_3cx_efficient_euler(decomposition, target_decomposed)
                elif best_nbasis == 2:
                    circuit = self._get_sx_vz_2cx_efficient_euler(decomposition, target_decomposed)
            else:
                raise QiskitError("pulse_optimizer currently only works with CNOT entangling gate")
        else:
            raise QiskitError(
                '"pulse_optimize" currently only works with ZSX basis '
                f"({self._decomposer1q.basis} used)"
            )
        return circuit

    def _get_sx_vz_2cx_efficient_euler(self, decomposition, target_decomposed):
        """
        Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
        two CNOT gates are needed.

        This first decomposes each unitary from the KAK decomposition into ZXZ on the source
        qubit of the CNOTs and XZX on the targets in order to commute operators to beginning and
        end of decomposition. The beginning and ending single qubit gates are then
        collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
        if performance is a concern.
        """
        best_nbasis = 2  # by assumption
        num_1q_uni = len(decomposition)
        # list of euler angle decompositions on qubits 0 and 1
        euler_q0 = np.empty((num_1q_uni // 2, 3), dtype=float)
        euler_q1 = np.empty((num_1q_uni // 2, 3), dtype=float)
        global_phase = 0.0

        # decompose source unitaries to zxz
        zxz_decomposer = OneQubitEulerDecomposer("ZXZ")
        for iqubit, decomp in enumerate(decomposition[0::2]):
            euler_angles = zxz_decomposer.angles_and_phase(decomp)
            euler_q0[iqubit, [1, 2, 0]] = euler_angles[:3]
            global_phase += euler_angles[3]
        # decompose target unitaries to xzx
        xzx_decomposer = OneQubitEulerDecomposer("XZX")
        for iqubit, decomp in enumerate(decomposition[1::2]):
            euler_angles = xzx_decomposer.angles_and_phase(decomp)
            euler_q1[iqubit, [1, 2, 0]] = euler_angles[:3]
            global_phase += euler_angles[3]
        qc = QuantumCircuit(2)
        qc.global_phase = target_decomposed.global_phase
        qc.global_phase -= best_nbasis * self.basis.global_phase
        qc.global_phase += global_phase

        # TODO: make this more effecient to avoid double decomposition
        # prepare beginning 0th qubit local unitary
        circ = QuantumCircuit(1)
        circ.rz(euler_q0[0][0], 0)
        circ.rx(euler_q0[0][1], 0)
        circ.rz(euler_q0[0][2] + euler_q0[1][0] + math.pi / 2, 0)
        # re-decompose to basis of 1q decomposer
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [0], inplace=True)

        # prepare beginning 1st qubit local unitary
        circ = QuantumCircuit(1)
        circ.rx(euler_q1[0][0], 0)
        circ.rz(euler_q1[0][1], 0)
        circ.rx(euler_q1[0][2] + euler_q1[1][0], 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [1], inplace=True)

        qc.cx(0, 1)
        # the central decompositions are dependent on the specific form of the
        # unitaries coming out of the two qubit decomposer which have some flexibility
        # of choice.
        qc.sx(0)
        qc.rz(euler_q0[1][1] - math.pi, 0)
        qc.sx(0)
        qc.rz(euler_q1[1][1], 1)
        qc.global_phase += math.pi / 2

        qc.cx(0, 1)

        circ = QuantumCircuit(1)
        circ.rz(euler_q0[1][2] + euler_q0[2][0] + math.pi / 2, 0)
        circ.rx(euler_q0[2][1], 0)
        circ.rz(euler_q0[2][2], 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [0], inplace=True)
        circ = QuantumCircuit(1)
        circ.rx(euler_q1[1][2] + euler_q1[2][0], 0)
        circ.rz(euler_q1[2][1], 0)
        circ.rx(euler_q1[2][2], 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [1], inplace=True)

        return qc

    def _get_sx_vz_3cx_efficient_euler(self, decomposition, target_decomposed):
        """
        Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
        three CNOT gates are needed.

        This first decomposes each unitary from the KAK decomposition into ZXZ on the source
        qubit of the CNOTs and XZX on the targets in order commute operators to beginning and
        end of decomposition. Inserting Hadamards reverses the direction of the CNOTs and transforms
        a variable Rx -> variable virtual Rz. The beginning and ending single qubit gates are then
        collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
        if performance is a concern.
        """
        best_nbasis = 3  # by assumption
        num_1q_uni = len(decomposition)
        # create structure to hold euler angles: 1st index represents unitary "group" wrt cx
        # 2nd index represents index of euler triple.
        euler_q0 = np.empty((num_1q_uni // 2, 3), dtype=float)
        euler_q1 = np.empty((num_1q_uni // 2, 3), dtype=float)
        global_phase = 0.0
        atol = 1e-10  # absolute tolerance for floats

        # decompose source unitaries to zxz
        zxz_decomposer = OneQubitEulerDecomposer("ZXZ")
        for iqubit, decomp in enumerate(decomposition[0::2]):
            euler_angles = zxz_decomposer.angles_and_phase(decomp)
            euler_q0[iqubit, [1, 2, 0]] = euler_angles[:3]
            global_phase += euler_angles[3]
        # decompose target unitaries to xzx
        xzx_decomposer = OneQubitEulerDecomposer("XZX")
        for iqubit, decomp in enumerate(decomposition[1::2]):
            euler_angles = xzx_decomposer.angles_and_phase(decomp)
            euler_q1[iqubit, [1, 2, 0]] = euler_angles[:3]
            global_phase += euler_angles[3]

        qc = QuantumCircuit(2)
        qc.global_phase = target_decomposed.global_phase
        qc.global_phase -= best_nbasis * self.basis.global_phase
        qc.global_phase += global_phase

        x12 = euler_q0[1][2] + euler_q0[2][0]
        x12_isNonZero = not math.isclose(x12, 0, abs_tol=atol)
        x12_isOddMult = None
        x12_isPiMult = math.isclose(math.sin(x12), 0, abs_tol=atol)
        if x12_isPiMult:
            x12_isOddMult = math.isclose(math.cos(x12), -1, abs_tol=atol)
            x12_phase = math.pi * math.cos(x12)
        x02_add = x12 - euler_q0[1][0]
        x12_isHalfPi = math.isclose(x12, math.pi / 2, abs_tol=atol)

        # TODO: make this more effecient to avoid double decomposition
        circ = QuantumCircuit(1)
        circ.rz(euler_q0[0][0], 0)
        circ.rx(euler_q0[0][1], 0)
        if x12_isNonZero and x12_isPiMult:
            circ.rz(euler_q0[0][2] - x02_add, 0)
        else:
            circ.rz(euler_q0[0][2] + euler_q0[1][0], 0)
        circ.h(0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [0], inplace=True)

        circ = QuantumCircuit(1)
        circ.rx(euler_q1[0][0], 0)
        circ.rz(euler_q1[0][1], 0)
        circ.rx(euler_q1[0][2] + euler_q1[1][0], 0)
        circ.h(0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [1], inplace=True)

        qc.cx(1, 0)

        if x12_isPiMult:
            # even or odd multiple
            if x12_isNonZero:
                qc.global_phase += x12_phase
            if x12_isNonZero and x12_isOddMult:
                qc.rz(-euler_q0[1][1], 0)
            else:
                qc.rz(euler_q0[1][1], 0)
                qc.global_phase += math.pi
        if x12_isHalfPi:
            qc.sx(0)
            qc.global_phase -= math.pi / 4
        elif x12_isNonZero and not x12_isPiMult:
            # this is non-optimal but doesn't seem to occur currently
            if self.pulse_optimize is None:
                qc.compose(self._decomposer1q(Operator(RXGate(x12)).data), [0], inplace=True)
            else:
                raise QiskitError("possible non-pulse-optimal decomposition encountered")
        if math.isclose(euler_q1[1][1], math.pi / 2, abs_tol=atol):
            qc.sx(1)
            qc.global_phase -= math.pi / 4
        else:
            # this is non-optimal but doesn't seem to occur currently
            if self.pulse_optimize is None:
                qc.compose(
                    self._decomposer1q(Operator(RXGate(euler_q1[1][1])).data), [1], inplace=True
                )
            else:
                raise QiskitError("possible non-pulse-optimal decomposition encountered")
        qc.rz(euler_q1[1][2] + euler_q1[2][0], 1)

        qc.cx(1, 0)

        qc.rz(euler_q0[2][1], 0)
        if math.isclose(euler_q1[2][1], math.pi / 2, abs_tol=atol):
            qc.sx(1)
            qc.global_phase -= math.pi / 4
        else:
            # this is non-optimal but doesn't seem to occur currently
            if self.pulse_optimize is None:
                qc.compose(
                    self._decomposer1q(Operator(RXGate(euler_q1[2][1])).data), [1], inplace=True
                )
            else:
                raise QiskitError("possible non-pulse-optimal decomposition encountered")

        qc.cx(1, 0)

        circ = QuantumCircuit(1)
        circ.h(0)
        circ.rz(euler_q0[2][2] + euler_q0[3][0], 0)
        circ.rx(euler_q0[3][1], 0)
        circ.rz(euler_q0[3][2], 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [0], inplace=True)

        circ = QuantumCircuit(1)
        circ.h(0)
        circ.rx(euler_q1[2][2] + euler_q1[3][0], 0)
        circ.rz(euler_q1[3][1], 0)
        circ.rx(euler_q1[3][2], 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [1], inplace=True)

        # TODO: fix the sign problem to avoid correction here
        if cmath.isclose(
            target_decomposed.unitary_matrix[0, 0], -(Operator(qc).data[0, 0]), abs_tol=atol
        ):
            qc.global_phase += math.pi
        return qc

    def num_basis_gates(self, unitary):
        """Computes the number of basis gates needed in
        a decomposition of input unitary
        """
        return two_qubit_decompose._num_basis_gates(
            self.basis.b, self.basis_fidelity, np.asarray(unitary, dtype=complex)
        )


class TwoQubitDecomposeUpToDiagonal:
    """
    Class to decompose two qubit unitaries into the product of a diagonal gate
    and another unitary gate which can be represented by two CX gates instead of the
    usual three. This can be used when neighboring gates commute with the diagonal to
    potentially reduce overall CX count.
    """

    def __init__(self):
        sy = np.array([[0, -1j], [1j, 0]])
        self.sysy = np.kron(sy, sy)

    def _u4_to_su4(self, u4):
        phase_factor = np.conj(np.linalg.det(u4) ** (-1 / u4.shape[0]))
        su4 = u4 / phase_factor
        return su4, cmath.phase(phase_factor)

    def _gamma(self, mat):
        """
        proposition II.1: this invariant characterizes when two operators in U(4),
        say u, v, are equivalent up to single qubit gates:

           u ≡ v -> Det(γ(u)) = Det(±(γ(v)))
        """
        sumat, _ = self._u4_to_su4(mat)
        sysy = self.sysy
        return sumat @ sysy @ sumat.T @ sysy

    def _cx0_test(self, mat):
        # proposition III.1: zero cx sufficient
        gamma = self._gamma(mat)
        evals = np.linalg.eigvals(gamma)
        return np.all(np.isclose(evals, np.ones(4)))

    def _cx1_test(self, mat):
        # proposition III.2: one cx sufficient
        gamma = self._gamma(mat)
        evals = np.linalg.eigvals(gamma)
        uvals, ucnts = np.unique(np.round(evals, 10), return_counts=True)
        return (
            len(uvals) == 2
            and all(ucnts == 2)
            and all((np.isclose(x, 1j)) or np.isclose(x, -1j) for x in uvals)
        )

    def _cx2_test(self, mat):
        # proposition III.3: two cx sufficient
        gamma = self._gamma(mat)
        return np.isclose(np.trace(gamma).imag, 0)

    def _real_trace_transform(self, mat):
        """
        Determine diagonal gate such that

        U3 = D U2

        Where U3 is a general two-qubit gate which takes 3 cnots, D is a
        diagonal gate, and U2 is a gate which takes 2 cnots.
        """
        a1 = (
            -mat[1, 3] * mat[2, 0]
            + mat[1, 2] * mat[2, 1]
            + mat[1, 1] * mat[2, 2]
            - mat[1, 0] * mat[2, 3]
        )
        a2 = (
            mat[0, 3] * mat[3, 0]
            - mat[0, 2] * mat[3, 1]
            - mat[0, 1] * mat[3, 2]
            + mat[0, 0] * mat[3, 3]
        )
        theta = 0  # arbitrary
        phi = 0  # arbitrary
        psi = np.arctan2(a1.imag + a2.imag, a1.real - a2.real) - phi
        diag = np.diag(np.exp(-1j * np.array([theta, phi, psi, -(theta + phi + psi)])))
        return diag

    def __call__(self, mat):
        """do the decomposition"""
        su4, phase = self._u4_to_su4(mat)
        real_map = self._real_trace_transform(su4)
        mapped_su4 = real_map @ su4
        if not self._cx2_test(mapped_su4):
            warnings.warn("Unitary decomposition up to diagonal may use an additionl CX gate.")
        circ = two_qubit_cnot_decompose(mapped_su4)
        circ.global_phase += phase
        return real_map.conj(), circ


# This weird duplicated lazy structure is for backwards compatibility; Qiskit has historically
# always made ``two_qubit_cnot_decompose`` available publicly immediately on import, but it's quite
# expensive to construct, and we want to defer the obejct's creation until it's actually used.  We
# only need to pass through the public methods that take `self` as a parameter.  Using `__getattr__`
# doesn't work because it is only called if the normal resolution methods fail.  Using
# `__getattribute__` is too messy for a simple one-off use object.


class _LazyTwoQubitCXDecomposer(TwoQubitBasisDecomposer):
    __slots__ = ("_inner",)

    def __init__(self):  # pylint: disable=super-init-not-called
        self._inner = None

    def _load(self):
        if self._inner is None:
            self._inner = TwoQubitBasisDecomposer(CXGate())

    def __call__(self, *args, **kwargs) -> QuantumCircuit:
        self._load()
        return self._inner(*args, **kwargs)

    def traces(self, target):
        self._load()
        return self._inner.traces(target)

    def decomp1(self, target):
        self._load()
        return self._inner.decomp1(target)

    def decomp2_supercontrolled(self, target):
        self._load()
        return self._inner.decomp2_supercontrolled(target)

    def decomp3_supercontrolled(self, target):
        self._load()
        return self._inner.decomp3_supercontrolled(target)

    def num_basis_gates(self, unitary):
        self._load()
        return self._inner.num_basis_gates(unitary)


two_qubit_cnot_decompose = _LazyTwoQubitCXDecomposer()
"""
This is an instance of :class:`.TwoQubitBasisDecomposer` that always uses
``cx`` as the KAK gate for the basis decomposition. You can use this function
as a quick access to ``cx``-based 2-qubit decompositions.

Args:
    unitary (Operator or np.ndarray): The 4x4 unitary to synthesize.
    basis_fidelity (float or None): If given the assumed fidelity for applications of :class:`.CXGate`.
    approximate (bool): If ``True`` approximate if ``basis_fidelity`` is less than 1.0.

Returns:
    QuantumCircuit: The synthesized circuit of the input unitary.
"""
