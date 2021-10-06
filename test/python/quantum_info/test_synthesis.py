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

"""Tests for quantum synthesis methods."""

import unittest
import contextlib
import logging
from test import combine

from ddt import ddt
import numpy as np

from qiskit import execute, QiskitError
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import (
    HGate,
    IGate,
    SdgGate,
    SGate,
    U3Gate,
    UGate,
    XGate,
    YGate,
    ZGate,
    CXGate,
    CZGate,
    iSwapGate,
    SwapGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    CPhaseGate,
    CRZGate,
    RXGate,
    RYGate,
    RZGate,
)
from qiskit.providers.basicaer import UnitarySimulatorPy
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import (
    TwoQubitWeylDecomposition,
    TwoQubitWeylIdEquiv,
    TwoQubitWeylSWAPEquiv,
    TwoQubitWeylPartialSWAPEquiv,
    TwoQubitWeylPartialSWAPFlipEquiv,
    TwoQubitWeylfSimaabEquiv,
    TwoQubitWeylfSimabbEquiv,
    TwoQubitWeylfSimabmbEquiv,
    TwoQubitWeylControlledEquiv,
    TwoQubitWeylMirrorControlledEquiv,
    TwoQubitWeylGeneral,
    two_qubit_cnot_decompose,
    TwoQubitBasisDecomposer,
    TwoQubitControlledUDecomposer,
    Ud,
    decompose_two_qubit_product_gate,
)

from qiskit.quantum_info.synthesis.ion_decompose import cnot_rxx_decompose
from qiskit.test import QiskitTestCase


def make_oneq_cliffords():
    """Make as list of 1q Cliffords"""
    ixyz_list = [g().to_matrix() for g in (IGate, XGate, YGate, ZGate)]
    ih_list = [g().to_matrix() for g in (IGate, HGate)]
    irs_list = [
        IGate().to_matrix(),
        SdgGate().to_matrix() @ HGate().to_matrix(),
        HGate().to_matrix() @ SGate().to_matrix(),
    ]
    oneq_cliffords = [
        Operator(ixyz @ ih @ irs) for ixyz in ixyz_list for ih in ih_list for irs in irs_list
    ]
    return oneq_cliffords


ONEQ_CLIFFORDS = make_oneq_cliffords()


def make_hard_thetas_oneq(smallest=1e-18, factor=3.2, steps=22, phi=0.7, lam=0.9):
    """Make 1q gates with theta/2 close to 0, pi/2, pi, 3pi/2"""
    return (
        [U3Gate(smallest * factor ** i, phi, lam) for i in range(steps)]
        + [U3Gate(-smallest * factor ** i, phi, lam) for i in range(steps)]
        + [U3Gate(np.pi / 2 + smallest * factor ** i, phi, lam) for i in range(steps)]
        + [U3Gate(np.pi / 2 - smallest * factor ** i, phi, lam) for i in range(steps)]
        + [U3Gate(np.pi + smallest * factor ** i, phi, lam) for i in range(steps)]
        + [U3Gate(np.pi - smallest * factor ** i, phi, lam) for i in range(steps)]
        + [U3Gate(3 * np.pi / 2 + smallest * factor ** i, phi, lam) for i in range(steps)]
        + [U3Gate(3 * np.pi / 2 - smallest * factor ** i, phi, lam) for i in range(steps)]
    )


HARD_THETA_ONEQS = make_hard_thetas_oneq()

# It's too slow to use all 24**4 Clifford combos. If we can make it faster, use a larger set
K1K2S = [
    (ONEQ_CLIFFORDS[3], ONEQ_CLIFFORDS[5], ONEQ_CLIFFORDS[2], ONEQ_CLIFFORDS[21]),
    (ONEQ_CLIFFORDS[5], ONEQ_CLIFFORDS[6], ONEQ_CLIFFORDS[9], ONEQ_CLIFFORDS[7]),
    (ONEQ_CLIFFORDS[2], ONEQ_CLIFFORDS[1], ONEQ_CLIFFORDS[0], ONEQ_CLIFFORDS[4]),
    [
        Operator(U3Gate(x, y, z))
        for x, y, z in [(0.2, 0.3, 0.1), (0.7, 0.15, 0.22), (0.001, 0.97, 2.2), (3.14, 2.1, 0.9)]
    ],
]


class CheckDecompositions(QiskitTestCase):
    """Implements decomposition checkers."""

    def check_one_qubit_euler_angles(self, operator, basis="U3", tolerance=1e-14, simplify=False):
        """Check OneQubitEulerDecomposer works for the given unitary"""
        target_unitary = operator.data
        if basis is None:
            angles = OneQubitEulerDecomposer().angles(target_unitary)
            decomp_unitary = U3Gate(*angles).to_matrix()
        else:
            decomposer = OneQubitEulerDecomposer(basis)
            decomp_unitary = Operator(decomposer(target_unitary, simplify=simplify)).data
        maxdist = np.max(np.abs(target_unitary - decomp_unitary))
        self.assertTrue(
            np.abs(maxdist) < tolerance, f"Operator {operator}: Worst distance {maxdist}"
        )

    @contextlib.contextmanager
    def assertDebugOnly(self):  # FIXME: when at python 3.10+ replace with assertNoLogs
        """Context manager, asserts log is emitted at level DEBUG but no higher"""
        with self.assertLogs("qiskit.quantum_info.synthesis", "DEBUG") as ctx:
            yield
        for i in range(len(ctx.records)):
            self.assertLessEqual(
                ctx.records[i].levelno,
                logging.DEBUG,
                msg=f"Unexpected logging entry: {ctx.output[i]}",
            )
            self.assertIn("Requested fidelity:", ctx.records[i].getMessage())

    def assertRoundTrip(self, weyl1: TwoQubitWeylDecomposition):
        """Fail if eval(repr(weyl1)) not equal to weyl1"""
        repr1 = repr(weyl1)
        with self.assertDebugOnly():
            weyl2: TwoQubitWeylDecomposition = eval(repr1)  # pylint: disable=eval-used
        msg_base = f"weyl1:\n{repr1}\nweyl2:\n{repr(weyl2)}"
        self.assertEqual(type(weyl1), type(weyl2), msg_base)
        maxdiff = np.max(abs(weyl1.unitary_matrix - weyl2.unitary_matrix))
        self.assertEqual(maxdiff, 0, msg=f"Unitary matrix differs by {maxdiff}\n" + msg_base)
        self.assertEqual(weyl1.a, weyl2.a, msg=msg_base)
        self.assertEqual(weyl1.b, weyl2.b, msg=msg_base)
        self.assertEqual(weyl1.c, weyl2.c, msg=msg_base)
        maxdiff = np.max(np.abs(weyl1.K1l - weyl2.K1l))
        self.assertEqual(maxdiff, 0, msg=f"K1l matrix differs by {maxdiff}" + msg_base)
        maxdiff = np.max(np.abs(weyl1.K1r - weyl2.K1r))
        self.assertEqual(maxdiff, 0, msg=f"K1r matrix differs by {maxdiff}" + msg_base)
        maxdiff = np.max(np.abs(weyl1.K2l - weyl2.K2l))
        self.assertEqual(maxdiff, 0, msg=f"K2l matrix differs by {maxdiff}" + msg_base)
        maxdiff = np.max(np.abs(weyl1.K2r - weyl2.K2r))
        self.assertEqual(maxdiff, 0, msg=f"K2r matrix differs by {maxdiff}" + msg_base)
        self.assertEqual(weyl1.requested_fidelity, weyl2.requested_fidelity, msg_base)

    def check_two_qubit_weyl_decomposition(self, target_unitary, tolerance=1.0e-12):
        """Check TwoQubitWeylDecomposition() works for a given operator"""
        # pylint: disable=invalid-name
        with self.assertDebugOnly():
            decomp = TwoQubitWeylDecomposition(target_unitary, fidelity=None)
        # self.assertRoundTrip(decomp)  # Too slow
        op = np.exp(1j * decomp.global_phase) * Operator(np.eye(4))
        for u, qs in (
            (decomp.K2r, [0]),
            (decomp.K2l, [1]),
            (Ud(decomp.a, decomp.b, decomp.c), [0, 1]),
            (decomp.K1r, [0]),
            (decomp.K1l, [1]),
        ):
            op = op.compose(u, qs)
        decomp_unitary = op.data
        maxdist = np.max(np.abs(target_unitary - decomp_unitary))
        self.assertLess(
            np.abs(maxdist),
            tolerance,
            f"{decomp}\nactual fid: {decomp.actual_fidelity()}\n"
            f"Unitary {target_unitary}:\nWorst distance {maxdist}",
        )

    def check_two_qubit_weyl_specialization(
        self, target_unitary, fidelity, expected_specialization, expected_gates
    ):
        """Check that the two qubit Weyl decomposition gets specialized as expected"""

        # Loop to check both for implicit and explicity specialization
        for decomposer in (TwoQubitWeylDecomposition, expected_specialization):
            with self.assertDebugOnly():
                decomp = decomposer(target_unitary, fidelity=fidelity)
            self.assertRoundTrip(decomp)
            self.assertEqual(
                np.max(np.abs(decomp.unitary_matrix - target_unitary)),
                0,
                "Incorrect saved unitary in the decomposition.",
            )
            self.assertIsInstance(decomp, expected_specialization, "Incorrect Weyl specialization.")
            circ = decomp.circuit(simplify=True)
            self.assertDictEqual(
                dict(circ.count_ops()), expected_gates, f"Gate counts of {decomposer.__name__}"
            )
            actual_fid = decomp.actual_fidelity()
            self.assertAlmostEqual(decomp.calculated_fidelity, actual_fid, places=13)
            self.assertGreaterEqual(actual_fid, fidelity, f"fidelity of {decomposer.__name__}")
            actual_unitary = Operator(circ).data
            trace = np.trace(actual_unitary.T.conj() @ target_unitary)
            self.assertAlmostEqual(
                trace.imag, 0, places=13, msg=f"Real trace for {decomposer.__name__}"
            )
        with self.assertDebugOnly():
            decomp2 = expected_specialization(target_unitary, fidelity=None)  # Shouldn't raise
        self.assertRoundTrip(decomp2)
        if expected_specialization is not TwoQubitWeylGeneral:
            with self.assertRaises(QiskitError) as exc:
                _ = expected_specialization(target_unitary, fidelity=1.0)
            self.assertIn("worse than requested", exc.exception.message)

    def check_exact_decomposition(
        self, target_unitary, decomposer, tolerance=1.0e-12, num_basis_uses=None
    ):
        """Check exact decomposition for a particular target"""
        decomp_circuit = decomposer(target_unitary, _num_basis_uses=num_basis_uses)
        if num_basis_uses is not None:
            self.assertEqual(num_basis_uses, decomp_circuit.count_ops().get("unitary", 0))
        result = execute(decomp_circuit, UnitarySimulatorPy(), optimization_level=0).result()
        decomp_unitary = result.get_unitary()
        maxdist = np.max(np.abs(target_unitary - decomp_unitary))
        self.assertTrue(
            np.abs(maxdist) < tolerance,
            f"Unitary {target_unitary}: Worst distance {maxdist}",
        )


@ddt
class TestEulerAngles1Q(CheckDecompositions):
    """Test euler_angles_1q()"""

    @combine(clifford=ONEQ_CLIFFORDS)
    def test_euler_angles_1q_clifford(self, clifford):
        """Verify euler_angles_1q produces correct Euler angles for all Cliffords."""
        self.check_one_qubit_euler_angles(clifford)

    @combine(gate=HARD_THETA_ONEQS)
    def test_euler_angles_1q_hard_thetas(self, gate):
        """Verify euler_angles_1q for close-to-degenerate theta"""
        self.check_one_qubit_euler_angles(Operator(gate))

    @combine(seed=range(5), name="test_euler_angles_1q_random_{seed}")
    def test_euler_angles_1q_random(self, seed):
        """Verify euler_angles_1q produces correct Euler angles for random_unitary (seed={seed})."""
        unitary = random_unitary(2, seed=seed)
        self.check_one_qubit_euler_angles(unitary)


ANGEXP_ZYZ = [
    [(1.0e-13, 0.1, -0.1, 0), (0, 0)],
    [(1.0e-13, 0.2, -0.1, 0), (1, 0)],
    [(1.0e-13, np.pi, np.pi, 0), (0, 0)],
    [(1.0e-13, np.pi, np.pi, np.pi), (0, 0)],
    [(np.pi, np.pi, np.pi, 0), (0, 1)],
    [(np.pi - 1.0e-13, np.pi, np.pi, np.pi), (0, 1)],
    [(np.pi, 0.1, 0.2, 0), (1, 1)],
    [(np.pi, 0.2, 0.2, 0), (0, 1)],
    [(1.0e-13, 0.1, 0.2, 0), (1, 0)],
    [(0.1, 0.2, 1.0e-13, 0), (1, 1)],
    [(0.1, 0.0, 0.0, 0), (0, 1)],
    [(0.1, 1.0e-13, 0.2, 0), (1, 1)],
    [(0.1, 0.2, 0.3, 0), (2, 1)],
    [(0.1, 0.2, np.pi, 0), (1, 1)],
    [(0.1, np.pi, 0.1, 0), (1, 1)],
    [(0.1, np.pi, np.pi, 0), (0, 1)],
]
"""
Special cases for ZYZ type expansions.  Each list entry is of the format

    (alpha, beta, gamma, delta), (r, s),

and encodes the assertion that

    (K(b) @ A(a) @ K(c), global_phase=d)

re-synthesizes to have r applications of the K gate and s of the A gate.
"""


ANGEXP_PSX = [
    [(0.0, 0.1, -0.1), (0, 0)],
    [(0.0, 0.1, 0.2), (1, 0)],
    [(-np.pi / 2, 0.2, 0.0), (2, 1)],
    [(np.pi / 2, 0.0, 0.21), (2, 1)],
    [(np.pi / 2, 0.12, 0.2), (2, 1)],
    [(np.pi / 2, -np.pi / 2, 0.21), (1, 1)],
    [(np.pi, np.pi, 0), (0, 2)],
    [(np.pi, np.pi + 0.1, 0.1), (0, 2)],
    [(np.pi, np.pi + 0.2, -0.1), (1, 2)],
    [(0.1, 0.2, 0.3), (3, 2)],
    [(0.1, np.pi, 0.2), (2, 2)],
    [(0.1, 0.2, 0.0), (2, 2)],
    [(0.1, 0.2, np.pi), (2, 2)],
    [(0.1, np.pi, 0), (1, 2)],
]
"""
Special cases for Z.X90.Z.X90.Z type expansions.  Each list entry is of the format

    (alpha, beta, gamma), (r, s),

and encodes the assertion that

    U3(alpha, beta, gamma)

re-synthesizes to have r applications of the P gate and s of the SX gate.
"""


@ddt
class TestOneQubitEulerSpecial(CheckDecompositions):
    """Test special cases for OneQubitEulerDecomposer.

    FIXME: Currently these are more like smoke tests that exercise each of the code paths
    and shapes of decompositions that can be made, but they don't check all the corner cases
    where a wrap by 2*pi might happen, etc
    """

    def check_oneq_special_cases(
        self,
        target,
        basis,
        expected_gates=None,
        tolerance=1.0e-12,
    ):
        """Check OneQubitEulerDecomposer produces the expected gates"""
        decomposer = OneQubitEulerDecomposer(basis)
        circ = decomposer(target, simplify=True)
        data = Operator(circ).data
        maxdist = np.max(np.abs(target.data - data))
        trace = np.trace(data.T.conj() @ target)
        self.assertLess(
            np.abs(maxdist),
            tolerance,
            f"Worst case distance: {maxdist}, trace: {trace}\n"
            f"Target:\n{target}\nActual:\n{data}\n{circ}",
        )
        if expected_gates is not None:
            self.assertDictEqual(dict(circ.count_ops()), expected_gates, f"Circuit:\n{circ}")

    @combine(angexp=ANGEXP_ZYZ)
    def test_special_ZYZ(self, angexp):
        """Special cases of ZYZ. {angexp[0]}"""
        a, b, c, d = angexp[0]
        exp = {("rz", "ry")[g]: angexp[1][g] for g in (0, 1) if angexp[1][g]}
        tgt = np.exp(1j * d) * RZGate(b).to_matrix() @ RYGate(a).to_matrix() @ RZGate(c).to_matrix()
        self.check_oneq_special_cases(tgt, "ZYZ", exp)

    @combine(angexp=ANGEXP_ZYZ)
    def test_special_ZXZ(self, angexp):
        """Special cases of ZXZ. {angexp[0]}"""
        a, b, c, d = angexp[0]
        exp = {("rz", "rx")[g]: angexp[1][g] for g in (0, 1) if angexp[1][g]}
        tgt = np.exp(1j * d) * RZGate(b).to_matrix() @ RXGate(a).to_matrix() @ RZGate(c).to_matrix()
        self.check_oneq_special_cases(tgt, "ZXZ", exp)

    @combine(angexp=ANGEXP_ZYZ)
    def test_special_XYX(self, angexp):
        """Special cases of XYX. {angexp[0]}"""
        a, b, c, d = angexp[0]
        exp = {("rx", "ry")[g]: angexp[1][g] for g in (0, 1) if angexp[1][g]}
        tgt = np.exp(1j * d) * RXGate(b).to_matrix() @ RYGate(a).to_matrix() @ RXGate(c).to_matrix()
        self.check_oneq_special_cases(tgt, "XYX", exp)

    def test_special_U321(self):
        """Special cases of U321"""
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, -0.1).to_matrix(), "U321", {})
        self.check_oneq_special_cases(U3Gate(0.0, 0.11, 0.2).to_matrix(), "U321", {"u1": 1})
        self.check_oneq_special_cases(U3Gate(np.pi / 2, 0.2, 0.0).to_matrix(), "U321", {"u2": 1})
        self.check_oneq_special_cases(U3Gate(np.pi / 2, 0.0, 0.2).to_matrix(), "U321", {"u2": 1})
        self.check_oneq_special_cases(U3Gate(0.11, 0.27, 0.3).to_matrix(), "U321", {"u3": 1})

    def test_special_U3(self):
        """Special cases of U3"""
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, -0.1).to_matrix(), "U3", {})
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, 0.2).to_matrix(), "U3", {"u3": 1})
        self.check_oneq_special_cases(U3Gate(np.pi / 2, 0.2, 0.0).to_matrix(), "U3", {"u3": 1})
        self.check_oneq_special_cases(U3Gate(np.pi / 2, 0.0, 0.2).to_matrix(), "U3", {"u3": 1})
        self.check_oneq_special_cases(U3Gate(0.11, 0.27, 0.3).to_matrix(), "U3", {"u3": 1})

    def test_special_U(self):
        """Special cases of U"""
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, -0.1).to_matrix(), "U", {})
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, 0.2).to_matrix(), "U", {"u": 1})
        self.check_oneq_special_cases(U3Gate(np.pi / 2, 0.2, 0.0).to_matrix(), "U", {"u": 1})
        self.check_oneq_special_cases(U3Gate(np.pi / 2, 0.0, 0.2).to_matrix(), "U", {"u": 1})
        self.check_oneq_special_cases(U3Gate(0.1, 0.2, 0.3).to_matrix(), "U", {"u": 1})

    def test_special_RR(self):
        """Special cases of RR"""
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, -0.1).to_matrix(), "RR", {})
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, 0.2).to_matrix(), "RR", {"r": 2})
        self.check_oneq_special_cases(U3Gate(-np.pi, 0.2, 0.0).to_matrix(), "RR", {"r": 1})
        self.check_oneq_special_cases(U3Gate(np.pi, 0.0, 0.2).to_matrix(), "RR", {"r": 1})
        self.check_oneq_special_cases(U3Gate(0.1, 0.2, 0.3).to_matrix(), "RR", {"r": 2})

    def test_special_U1X(self):
        """Special cases of U1X"""
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, -0.1).to_matrix(), "U1X", {})
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, 0.2).to_matrix(), "U1X", {"u1": 1})
        self.check_oneq_special_cases(
            U3Gate(-np.pi / 2, 0.2, 0.0).to_matrix(), "U1X", {"u1": 2, "rx": 1}
        )
        self.check_oneq_special_cases(
            U3Gate(np.pi / 2, 0.0, 0.21).to_matrix(), "U1X", {"u1": 2, "rx": 1}
        )
        self.check_oneq_special_cases(
            U3Gate(np.pi / 2, 0.12, 0.2).to_matrix(), "U1X", {"u1": 2, "rx": 1}
        )
        self.check_oneq_special_cases(U3Gate(0.1, 0.2, 0.3).to_matrix(), "U1X", {"u1": 3, "rx": 2})

    @combine(angexp=ANGEXP_PSX)
    def test_special_PSX(self, angexp):
        """Special cases of PSX. {angexp[0]}"""
        a, b, c = angexp[0]
        tgt = U3Gate(a, b, c).to_matrix()
        exp = {("p", "sx")[g]: angexp[1][g] for g in (0, 1) if angexp[1][g]}
        self.check_oneq_special_cases(tgt, "PSX", exp)

    def test_special_ZSX(self):
        """Special cases of ZSX"""
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, -0.1).to_matrix(), "ZSX", {})
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, 0.2).to_matrix(), "ZSX", {"rz": 1})
        self.check_oneq_special_cases(
            U3Gate(-np.pi / 2, 0.2, 0.0).to_matrix(), "ZSX", {"rz": 2, "sx": 1}
        )
        self.check_oneq_special_cases(
            U3Gate(np.pi / 2, 0.0, 0.21).to_matrix(), "ZSX", {"rz": 2, "sx": 1}
        )
        self.check_oneq_special_cases(
            U3Gate(np.pi / 2, 0.12, 0.2).to_matrix(), "ZSX", {"rz": 2, "sx": 1}
        )
        self.check_oneq_special_cases(U3Gate(0.1, 0.2, 0.3).to_matrix(), "ZSX", {"rz": 3, "sx": 2})

    def test_special_ZSXX(self):
        """Special cases of ZSXX"""
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, -0.1).to_matrix(), "ZSXX", {})
        self.check_oneq_special_cases(U3Gate(0.0, 0.1, 0.2).to_matrix(), "ZSXX", {"rz": 1})
        self.check_oneq_special_cases(
            U3Gate(-np.pi / 2, 0.2, 0.0).to_matrix(), "ZSXX", {"rz": 2, "sx": 1}
        )
        self.check_oneq_special_cases(
            U3Gate(np.pi / 2, 0.0, 0.21).to_matrix(), "ZSXX", {"rz": 2, "sx": 1}
        )
        self.check_oneq_special_cases(
            U3Gate(np.pi / 2, 0.12, 0.2).to_matrix(), "ZSXX", {"rz": 2, "sx": 1}
        )
        self.check_oneq_special_cases(U3Gate(0.1, 0.2, 0.3).to_matrix(), "ZSXX", {"rz": 3, "sx": 2})
        self.check_oneq_special_cases(
            U3Gate(np.pi, 0.2, 0.3).to_matrix(), "ZSXX", {"rz": 1, "x": 1}
        )
        self.check_oneq_special_cases(
            U3Gate(np.pi, -np.pi / 2, np.pi / 2).to_matrix(), "ZSXX", {"x": 1}
        )


ONEQ_BASES = ["U3", "U321", "U", "U1X", "PSX", "ZSX", "ZSXX", "ZYZ", "ZXZ", "XYX", "RR", "XZX"]
SIMP_TOL = [
    (False, 1.0e-14),
    (True, 1.0e-12),
]  # Please don't broaden the tolerance (fix the decomp)


@ddt
class TestOneQubitEulerDecomposer(CheckDecompositions):
    """Test OneQubitEulerDecomposer"""

    @combine(
        basis=ONEQ_BASES,
        simp_tol=SIMP_TOL,
        name="test_one_qubit_clifford_{basis}_basis_simplify_{simp_tol[0]}",
    )
    def test_one_qubit_clifford_all_basis(self, basis, simp_tol):
        """Verify for {basis} basis and all Cliffords."""
        for clifford in ONEQ_CLIFFORDS:
            self.check_one_qubit_euler_angles(
                clifford, basis, simplify=simp_tol[0], tolerance=simp_tol[1]
            )

    @combine(
        basis=ONEQ_BASES,
        simp_tol=SIMP_TOL,
        name="test_one_qubit_hard_thetas_{basis}_basis_simplify_{simp_tol[0]}",
    )
    def test_one_qubit_hard_thetas_all_basis(self, basis, simp_tol):
        """Verify for {basis} basis and close-to-degenerate theta."""
        for gate in HARD_THETA_ONEQS:
            self.check_one_qubit_euler_angles(
                Operator(gate), basis, simplify=simp_tol[0], tolerance=simp_tol[1]
            )

    @combine(
        basis=ONEQ_BASES,
        simp_tol=SIMP_TOL,
        seed=range(50),
        name="test_one_qubit_random_{basis}_basis_simplify_{simp_tol[0]}_{seed}",
    )
    def test_one_qubit_random_all_basis(self, basis, simp_tol, seed):
        """Verify for {basis} basis and random_unitary (seed={seed})."""
        unitary = random_unitary(2, seed=seed)
        self.check_one_qubit_euler_angles(
            unitary, basis, simplify=simp_tol[0], tolerance=simp_tol[1]
        )

    def test_psx_zsx_special_cases(self):
        """Test decompositions of psx and zsx at special values of parameters"""
        oqed_psx = OneQubitEulerDecomposer(basis="PSX")
        oqed_zsx = OneQubitEulerDecomposer(basis="ZSX")
        oqed_zsxx = OneQubitEulerDecomposer(basis="ZSXX")
        theta = np.pi / 3
        phi = np.pi / 5
        lam = np.pi / 7
        test_gates = [
            UGate(np.pi, phi, lam),
            UGate(-np.pi, phi, lam),
            # test abs(lam + phi + theta) near 0
            UGate(np.pi, np.pi / 3, 2 * np.pi / 3),
            # test theta=pi/2
            UGate(np.pi / 2, phi, lam),
            # test theta=pi/2 and theta+lam=0
            UGate(np.pi / 2, phi, -np.pi / 2),
            # test theta close to 3*pi/2 and theta+phi=2*pi
            UGate(3 * np.pi / 2, np.pi / 2, lam),
            # test theta 0
            UGate(0, phi, lam),
            # test phi 0
            UGate(theta, 0, lam),
            # test lam 0
            UGate(theta, phi, 0),
        ]

        for gate in test_gates:
            unitary = gate.to_matrix()
            qc_psx = oqed_psx(unitary)
            qc_zsx = oqed_zsx(unitary)
            qc_zsxx = oqed_zsxx(unitary)
            self.assertTrue(np.allclose(unitary, Operator(qc_psx).data))
            self.assertTrue(np.allclose(unitary, Operator(qc_zsx).data))
            self.assertTrue(np.allclose(unitary, Operator(qc_zsxx).data))


# FIXME: streamline the set of test cases
class TestTwoQubitWeylDecomposition(CheckDecompositions):
    """Test TwoQubitWeylDecomposition()"""

    def test_TwoQubitWeylDecomposition_repr(self, seed=42):
        """Check that eval(__repr__) is exact round trip"""
        target = random_unitary(4, seed=seed)
        weyl1 = TwoQubitWeylDecomposition(target, fidelity=0.99)
        self.assertRoundTrip(weyl1)

    def test_two_qubit_weyl_decomposition_cnot(self):
        """Verify Weyl KAK decomposition for U~CNOT"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi / 4, 0, 0)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_iswap(self):
        """Verify Weyl KAK decomposition for U~iswap"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi / 4, np.pi / 4, 0)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_swap(self):
        """Verify Weyl KAK decomposition for U~swap"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi / 4, np.pi / 4, np.pi / 4)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_bgate(self):
        """Verify Weyl KAK decomposition for U~B"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi / 4, np.pi / 8, 0)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_a00(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,0,0)"""
        for aaa in (
            [smallest * factor ** i for i in range(steps)]
            + [np.pi / 4 - smallest * factor ** i for i in range(steps)]
            + [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]
        ):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, 0, 0)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aa0(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,0)"""
        for aaa in (
            [smallest * factor ** i for i in range(steps)]
            + [np.pi / 4 - smallest * factor ** i for i in range(steps)]
            + [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]
        ):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, aaa, 0)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aaa(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,a)"""
        for aaa in (
            [smallest * factor ** i for i in range(steps)]
            + [np.pi / 4 - smallest * factor ** i for i in range(steps)]
            + [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]
        ):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, aaa, aaa)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aama(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,-a)"""
        for aaa in (
            [smallest * factor ** i for i in range(steps)]
            + [np.pi / 4 - smallest * factor ** i for i in range(steps)]
            + [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]
        ):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, aaa, -aaa)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_ab0(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,b,0)"""
        for aaa in (
            [smallest * factor ** i for i in range(steps)]
            + [np.pi / 4 - smallest * factor ** i for i in range(steps)]
            + [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]
        ):
            for bbb in np.linspace(0, aaa, 10):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, bbb, 0)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_abb(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,b,b)"""
        for aaa in (
            [smallest * factor ** i for i in range(steps)]
            + [np.pi / 4 - smallest * factor ** i for i in range(steps)]
            + [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]
        ):
            for bbb in np.linspace(0, aaa, 6):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, bbb, bbb)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_abmb(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,b,-b)"""
        for aaa in (
            [smallest * factor ** i for i in range(steps)]
            + [np.pi / 4 - smallest * factor ** i for i in range(steps)]
            + [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]
        ):
            for bbb in np.linspace(0, aaa, 6):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, bbb, -bbb)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aac(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,c)"""
        for aaa in (
            [smallest * factor ** i for i in range(steps)]
            + [np.pi / 4 - smallest * factor ** i for i in range(steps)]
            + [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]
        ):
            for ccc in np.linspace(-aaa, aaa, 6):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, aaa, ccc)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_abc(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,b,c)"""
        for aaa in (
            [smallest * factor ** i for i in range(steps)]
            + [np.pi / 4 - smallest * factor ** i for i in range(steps)]
            + [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]
        ):
            for bbb in np.linspace(0, aaa, 4):
                for ccc in np.linspace(-bbb, bbb, 4):
                    for k1l, k1r, k2l, k2r in K1K2S:
                        k1 = np.kron(k1l.data, k1r.data)
                        k2 = np.kron(k2l.data, k2r.data)
                        a = Ud(aaa, bbb, ccc)
                        self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)


K1K2SB = [
    [Operator(U3Gate(*xyz)) for xyz in xyzs]
    for xyzs in [
        [(0.2, 0.3, 0.1), (0.7, 0.15, 0.22), (0.1, 0.97, 2.2), (3.14, 2.1, 0.9)],
        [(0.21, 0.13, 0.45), (2.1, 0.77, 0.88), (1.5, 2.3, 2.3), (2.1, 0.4, 1.7)],
    ]
]
DELTAS = [
    (-0.019, 0.018, 0.021),
    (0.01, 0.015, 0.02),
    (-0.01, -0.009, 0.011),
    (-0.002, -0.003, -0.004),
]


class TestTwoQubitWeylDecompositionSpecialization(CheckDecompositions):
    """Check TwoQubitWeylDecomposition specialized subclasses"""

    def test_weyl_specialize_id(self):
        """Weyl specialization for Id gate"""
        a, b, c = 0.0, 0.0, 0.0
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylIdEquiv,
                    {"rz": 4, "ry": 2},
                )

    def test_weyl_specialize_swap(self):
        """Weyl specialization for swap gate"""
        a, b, c = np.pi / 4, np.pi / 4, np.pi / 4
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylSWAPEquiv,
                    {"rz": 4, "ry": 2, "swap": 1},
                )

    def test_weyl_specialize_flip_swap(self):
        """Weyl specialization for flip swap gate"""
        a, b, c = np.pi / 4, np.pi / 4, -np.pi / 4
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylSWAPEquiv,
                    {"rz": 4, "ry": 2, "swap": 1},
                )

    def test_weyl_specialize_pswap(self, theta=0.123):
        """Weyl specialization for partial swap gate"""
        a, b, c = theta, theta, theta
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylPartialSWAPEquiv,
                    {"rz": 6, "ry": 3, "rxx": 1, "ryy": 1, "rzz": 1},
                )

    def test_weyl_specialize_flip_pswap(self, theta=0.123):
        """Weyl specialization for flipped partial swap gate"""
        a, b, c = theta, theta, -theta
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylPartialSWAPFlipEquiv,
                    {"rz": 6, "ry": 3, "rxx": 1, "ryy": 1, "rzz": 1},
                )

    def test_weyl_specialize_fsim_aab(self, aaa=0.456, bbb=0.132):
        """Weyl specialization for partial swap gate"""
        a, b, c = aaa, aaa, bbb
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylfSimaabEquiv,
                    {"rz": 7, "ry": 4, "rxx": 1, "ryy": 1, "rzz": 1},
                )

    def test_weyl_specialize_fsim_abb(self, aaa=0.456, bbb=0.132):
        """Weyl specialization for partial swap gate"""
        a, b, c = aaa, bbb, bbb
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylfSimabbEquiv,
                    {"rx": 7, "ry": 4, "rxx": 1, "ryy": 1, "rzz": 1},
                )

    def test_weyl_specialize_fsim_abmb(self, aaa=0.456, bbb=0.132):
        """Weyl specialization for partial swap gate"""
        a, b, c = aaa, bbb, -bbb
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylfSimabmbEquiv,
                    {"rx": 7, "ry": 4, "rxx": 1, "ryy": 1, "rzz": 1},
                )

    def test_weyl_specialize_ctrl(self, aaa=0.456):
        """Weyl specialization for partial swap gate"""
        a, b, c = aaa, 0.0, 0.0
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylControlledEquiv,
                    {"rx": 6, "ry": 4, "rxx": 1},
                )

    def test_weyl_specialize_mirror_ctrl(self, aaa=-0.456):
        """Weyl specialization for partial swap gate"""
        a, b, c = np.pi / 4, np.pi / 4, aaa
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylMirrorControlledEquiv,
                    {"rz": 6, "ry": 4, "rzz": 1, "swap": 1},
                )

    def test_weyl_specialize_general(self, aaa=0.456, bbb=0.345, ccc=0.123):
        """Weyl specialization for partial swap gate"""
        a, b, c = aaa, bbb, ccc
        for da, db, dc in DELTAS:
            for k1l, k1r, k2l, k2r in K1K2SB:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                self.check_two_qubit_weyl_specialization(
                    k1 @ Ud(a + da, b + db, c + dc) @ k2,
                    0.999,
                    TwoQubitWeylGeneral,
                    {"rz": 8, "ry": 4, "rxx": 1, "ryy": 1, "rzz": 1},
                )


@ddt
class TestTwoQubitDecompose(CheckDecompositions):
    """Test TwoQubitBasisDecomposer() for exact/approx decompositions"""

    def test_cnot_rxx_decompose(self):
        """Verify CNOT decomposition into RXX gate is correct"""
        cnot = Operator(CXGate())
        decomps = [
            cnot_rxx_decompose(),
            cnot_rxx_decompose(plus_ry=True, plus_rxx=True),
            cnot_rxx_decompose(plus_ry=True, plus_rxx=False),
            cnot_rxx_decompose(plus_ry=False, plus_rxx=True),
            cnot_rxx_decompose(plus_ry=False, plus_rxx=False),
        ]
        for decomp in decomps:
            self.assertTrue(cnot.equiv(decomp))

    @combine(seed=range(10), name="test_exact_two_qubit_cnot_decompose_random_{seed}")
    def test_exact_two_qubit_cnot_decompose_random(self, seed):
        """Verify exact CNOT decomposition for random Haar 4x4 unitary (seed={seed})."""
        unitary = random_unitary(4, seed=seed)
        self.check_exact_decomposition(unitary.data, two_qubit_cnot_decompose)

    def test_exact_two_qubit_cnot_decompose_paulis(self):
        """Verify exact CNOT decomposition for Paulis"""
        unitary = Operator.from_label("XZ")
        self.check_exact_decomposition(unitary.data, two_qubit_cnot_decompose)

    def make_random_supercontrolled_decomposer(self, seed):
        """Return a random supercontrolled unitary given a seed"""
        state = np.random.default_rng(seed)
        basis_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_phase = state.random() * 2 * np.pi
        basis_b = state.random() * np.pi / 4
        basis_unitary = np.exp(1j * basis_phase) * basis_k1 @ Ud(np.pi / 4, basis_b, 0) @ basis_k2
        decomposer = TwoQubitBasisDecomposer(UnitaryGate(basis_unitary))
        return decomposer

    @combine(seed=range(10), name="test_exact_supercontrolled_decompose_random_{seed}")
    def test_exact_supercontrolled_decompose_random(self, seed):
        """Exact decomposition for random supercontrolled basis and random target (seed={seed})"""
        # pylint: disable=invalid-name
        state = np.random.default_rng(seed)
        decomposer = self.make_random_supercontrolled_decomposer(state)
        self.check_exact_decomposition(random_unitary(4, seed=state).data, decomposer)

    @combine(seed=range(10), name="seed_{seed}")
    def test_exact_supercontrolled_decompose_phase_0_use_random(self, seed):
        """Exact decomposition supercontrolled basis, random target (0 basis uses) (seed={seed})"""
        state = np.random.default_rng(seed)
        decomposer = self.make_random_supercontrolled_decomposer(state)

        tgt_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_phase = state.random() * 2 * np.pi
        tgt_unitary = np.exp(1j * tgt_phase) * tgt_k1 @ Ud(0, 0, 0) @ tgt_k2
        self.check_exact_decomposition(tgt_unitary, decomposer, num_basis_uses=0)

    @combine(seed=range(10), name="seed_{seed}")
    def test_exact_supercontrolled_decompose_phase_1_use_random(self, seed):
        """Exact decomposition supercontrolled basis, random tgt (1 basis uses) (seed={seed})"""
        state = np.random.default_rng(seed)
        basis_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_phase = state.random() * 2 * np.pi
        basis_b = state.random() * np.pi / 4
        basis_unitary = np.exp(1j * basis_phase) * basis_k1 @ Ud(np.pi / 4, basis_b, 0) @ basis_k2
        decomposer = TwoQubitBasisDecomposer(UnitaryGate(basis_unitary))

        tgt_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_phase = state.random() * 2 * np.pi

        tgt_unitary = np.exp(1j * tgt_phase) * tgt_k1 @ Ud(np.pi / 4, basis_b, 0) @ tgt_k2
        self.check_exact_decomposition(tgt_unitary, decomposer, num_basis_uses=1)

    @combine(seed=range(10), name="seed_{seed}")
    def test_exact_supercontrolled_decompose_phase_2_use_random(self, seed):
        """Exact decomposition supercontrolled basis, random tgt (2 basis uses) (seed={seed})"""
        state = np.random.default_rng(seed)
        decomposer = self.make_random_supercontrolled_decomposer(state)

        tgt_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_phase = state.random() * 2 * np.pi
        tgt_a, tgt_b = state.random(size=2) * np.pi / 4
        tgt_unitary = np.exp(1j * tgt_phase) * tgt_k1 @ Ud(tgt_a, tgt_b, 0) @ tgt_k2
        self.check_exact_decomposition(tgt_unitary, decomposer, num_basis_uses=2)

    @combine(seed=range(10), name="seed_{seed}")
    def test_exact_supercontrolled_decompose_phase_3_use_random(self, seed):
        """Exact decomposition supercontrolled basis, random tgt (3 basis uses) (seed={seed})"""
        state = np.random.default_rng(seed)
        decomposer = self.make_random_supercontrolled_decomposer(state)

        tgt_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_phase = state.random() * 2 * np.pi

        tgt_a, tgt_b = state.random(size=2) * np.pi / 4
        tgt_c = state.random() * np.pi / 2 - np.pi / 4
        tgt_unitary = np.exp(1j * tgt_phase) * tgt_k1 @ Ud(tgt_a, tgt_b, tgt_c) @ tgt_k2
        self.check_exact_decomposition(tgt_unitary, decomposer, num_basis_uses=3)

    def test_exact_nonsupercontrolled_decompose(self):
        """Check that the nonsupercontrolled basis throws a warning"""
        with self.assertWarns(UserWarning, msg="Supposed to warn when basis non-supercontrolled"):
            TwoQubitBasisDecomposer(UnitaryGate(Ud(np.pi / 4, 0.2, 0.1)))

    @combine(seed=range(10), name="seed_{seed}")
    def test_approx_supercontrolled_decompose_random(self, seed):
        """Check that n-uses of supercontrolled basis give the expected trace distance"""
        state = np.random.default_rng(seed)
        decomposer = self.make_random_supercontrolled_decomposer(state)

        tgt_phase = state.random() * 2 * np.pi
        tgt = random_unitary(4, seed=state).data
        tgt *= np.exp(1j * tgt_phase)

        with self.assertDebugOnly():
            traces_pred = decomposer.traces(TwoQubitWeylDecomposition(tgt))

        for i in range(4):
            with self.subTest(i=i):
                decomp_circuit = decomposer(tgt, _num_basis_uses=i)
                result = execute(
                    decomp_circuit, UnitarySimulatorPy(), optimization_level=0
                ).result()
                decomp_unitary = result.get_unitary()
                tr_actual = np.trace(decomp_unitary.conj().T @ tgt)
                self.assertAlmostEqual(
                    traces_pred[i],
                    tr_actual,
                    places=13,
                    msg=f"Trace doesn't match for {i}-basis decomposition",
                )

    def test_cx_equivalence_0cx(self, seed=0):
        """Check circuits with  0 cx gates locally equivalent to identity"""
        state = np.random.default_rng(seed)
        rnd = 2 * np.pi * state.random(size=6)

        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        sim = UnitarySimulatorPy()
        unitary = execute(qc, sim).result().get_unitary()
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 0)
        self.assertTrue(Operator(two_qubit_cnot_decompose(unitary)).equiv(unitary))

    def test_cx_equivalence_1cx(self, seed=1):
        """Check circuits with  1 cx gates locally equivalent to a cx"""
        state = np.random.default_rng(seed)
        rnd = 2 * np.pi * state.random(size=12)

        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[6], rnd[7], rnd[8], qr[0])
        qc.u(rnd[9], rnd[10], rnd[11], qr[1])

        sim = UnitarySimulatorPy()
        unitary = execute(qc, sim).result().get_unitary()
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 1)
        self.assertTrue(Operator(two_qubit_cnot_decompose(unitary)).equiv(unitary))

    def test_cx_equivalence_2cx(self, seed=2):
        """Check circuits with  2 cx gates locally equivalent to some circuit with 2 cx."""
        state = np.random.default_rng(seed)
        rnd = 2 * np.pi * state.random(size=18)

        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[6], rnd[7], rnd[8], qr[0])
        qc.u(rnd[9], rnd[10], rnd[11], qr[1])

        qc.cx(qr[0], qr[1])

        qc.u(rnd[12], rnd[13], rnd[14], qr[0])
        qc.u(rnd[15], rnd[16], rnd[17], qr[1])

        sim = UnitarySimulatorPy()
        unitary = execute(qc, sim).result().get_unitary()
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 2)
        self.assertTrue(Operator(two_qubit_cnot_decompose(unitary)).equiv(unitary))

    def test_cx_equivalence_3cx(self, seed=3):
        """Check circuits with 3 cx gates are outside the 0, 1, and 2 qubit regions."""
        state = np.random.default_rng(seed)
        rnd = 2 * np.pi * state.random(size=24)

        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[6], rnd[7], rnd[8], qr[0])
        qc.u(rnd[9], rnd[10], rnd[11], qr[1])

        qc.cx(qr[0], qr[1])

        qc.u(rnd[12], rnd[13], rnd[14], qr[0])
        qc.u(rnd[15], rnd[16], rnd[17], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[18], rnd[19], rnd[20], qr[0])
        qc.u(rnd[21], rnd[22], rnd[23], qr[1])

        sim = UnitarySimulatorPy()
        unitary = execute(qc, sim).result().get_unitary()
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 3)
        self.assertTrue(Operator(two_qubit_cnot_decompose(unitary)).equiv(unitary))

    def test_seed_289(self):
        """This specific case failed when PR #3585 was applied
        See https://github.com/Qiskit/qiskit-terra/pull/3652"""
        unitary = random_unitary(4, seed=289)
        self.check_exact_decomposition(unitary.data, two_qubit_cnot_decompose)

    @combine(
        seed=range(10),
        euler_bases=[
            ("U321", ["u3", "u2", "u1"]),
            ("U3", ["u3"]),
            ("U", ["u"]),
            ("U1X", ["u1", "rx"]),
            ("RR", ["r"]),
            ("PSX", ["p", "sx"]),
            ("ZYZ", ["rz", "ry"]),
            ("ZXZ", ["rz", "rx"]),
            ("XYX", ["rx", "ry"]),
            ("ZSX", ["rz", "sx"]),
            ("ZSXX", ["rz", "sx", "x"]),
        ],
        kak_gates=[
            (CXGate(), "cx"),
            (CZGate(), "cz"),
            (iSwapGate(), "iswap"),
            (RXXGate(np.pi / 2), "rxx"),
        ],
        name="test_euler_basis_selection_{seed}_{euler_bases[0]}_{kak_gates[1]}",
    )
    def test_euler_basis_selection(self, euler_bases, kak_gates, seed):
        """Verify decomposition uses euler_basis for 1q gates."""
        (euler_basis, oneq_gates) = euler_bases
        (kak_gate, kak_gate_name) = kak_gates

        with self.subTest(euler_basis=euler_basis, kak_gate=kak_gate):
            decomposer = TwoQubitBasisDecomposer(kak_gate, euler_basis=euler_basis)
            unitary = random_unitary(4, seed=seed)
            self.check_exact_decomposition(unitary.data, decomposer)

            decomposition_basis = set(decomposer(unitary).count_ops())
            requested_basis = set(oneq_gates + [kak_gate_name])
            self.assertTrue(decomposition_basis.issubset(requested_basis))


@ddt
class TestPulseOptimalDecompose(CheckDecompositions):
    """Check pulse optimal decomposition."""

    @combine(seed=range(10), name="seed_{seed}")
    def test_sx_virtz_3cnot_optimal(self, seed):
        """Test 3 CNOT ZSX pulse optimal decomposition"""
        unitary = random_unitary(4, seed=seed)
        decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX", pulse_optimize=True)
        circ = decomposer(unitary)
        self.assertEqual(Operator(unitary), Operator(circ))
        self.assertEqual(self._remove_pre_post_1q(circ).count_ops().get("sx"), 2)

    @combine(seed=range(10), name="seed_{seed}")
    def test_sx_virtz_2cnot_optimal(self, seed):
        """Test 2 CNOT ZSX pulse optimal decomposition"""
        rng = np.random.default_rng(seed)
        decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX", pulse_optimize=True)
        tgt_k1 = np.kron(random_unitary(2, seed=rng).data, random_unitary(2, seed=rng).data)
        tgt_k2 = np.kron(random_unitary(2, seed=rng).data, random_unitary(2, seed=rng).data)
        tgt_phase = rng.random() * 2 * np.pi
        tgt_a, tgt_b = rng.random(size=2) * np.pi / 4
        tgt_unitary = np.exp(1j * tgt_phase) * tgt_k1 @ Ud(tgt_a, tgt_b, 0) @ tgt_k2
        circ = decomposer(tgt_unitary)
        self.assertEqual(Operator(tgt_unitary), Operator(circ))

    def _remove_pre_post_1q(self, circ):
        """remove single qubit operations before and after all multi-qubit ops"""
        dag = circuit_to_dag(circ)
        del_list = []
        for node in dag.topological_op_nodes():
            if len(node.qargs) > 1:
                break
            del_list.append(node)
        for node in reversed(list(dag.topological_op_nodes())):
            if len(node.qargs) > 1:
                break
            del_list.append(node)
        for node in del_list:
            dag.remove_op_node(node)
        return dag_to_circuit(dag)


@ddt
class TestTwoQubitDecomposeApprox(CheckDecompositions):
    """Smoke tests for automatically-chosen approximate decompositions"""

    def check_approx_decomposition(self, target_unitary, decomposer, num_basis_uses):
        """Check approx decomposition for a particular target"""
        self.assertEqual(decomposer.num_basis_gates(target_unitary), num_basis_uses)
        decomp_circuit = decomposer(target_unitary)
        self.assertEqual(num_basis_uses, decomp_circuit.count_ops().get("unitary", 0))

    @combine(seed=range(10), name="seed_{seed}")
    def test_approx_supercontrolled_decompose_phase_0_use_random(self, seed, delta=0.01):
        """Approx decomposition supercontrolled basis, random target (0 basis uses) (seed={seed})"""
        state = np.random.default_rng(seed)
        basis_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_phase = state.random() * 2 * np.pi
        basis_b = 0.4  # how to safely randomize?
        basis_unitary = np.exp(1j * basis_phase) * basis_k1 @ Ud(np.pi / 4, basis_b, 0) @ basis_k2
        decomposer = TwoQubitBasisDecomposer(UnitaryGate(basis_unitary), basis_fidelity=0.99)

        tgt_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_phase = state.random() * 2 * np.pi
        d1, d2, d3 = state.random(size=3) * delta
        tgt_unitary = np.exp(1j * tgt_phase) * tgt_k1 @ Ud(d1, d2, d3) @ tgt_k2
        self.check_approx_decomposition(tgt_unitary, decomposer, num_basis_uses=0)

    @combine(seed=range(10), name="seed_{seed}")
    def test_approx_supercontrolled_decompose_phase_1_use_random(self, seed, delta=0.01):
        """Approximate decomposition supercontrolled basis, random tgt (1 basis uses) (seed={seed})"""
        state = np.random.default_rng(seed)
        basis_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_phase = state.random() * 2 * np.pi
        basis_b = 0.4  # how to safely randomize?
        basis_unitary = np.exp(1j * basis_phase) * basis_k1 @ Ud(np.pi / 4, basis_b, 0) @ basis_k2
        decomposer = TwoQubitBasisDecomposer(UnitaryGate(basis_unitary), basis_fidelity=0.99)

        tgt_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_phase = state.random() * 2 * np.pi
        d1, d2, d3 = state.random(size=3) * delta
        tgt_unitary = (
            np.exp(1j * tgt_phase) * tgt_k1 @ Ud(np.pi / 4 - d1, basis_b + d2, d3) @ tgt_k2
        )
        self.check_approx_decomposition(tgt_unitary, decomposer, num_basis_uses=1)

    @combine(seed=range(10), name="seed_{seed}")
    def test_approx_supercontrolled_decompose_phase_2_use_random(self, seed, delta=0.01):
        """Approximate decomposition supercontrolled basis, random tgt (2 basis uses) (seed={seed})"""
        state = np.random.default_rng(seed)
        basis_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_phase = state.random() * 2 * np.pi
        basis_b = 0.4  # how to safely randomize?
        basis_unitary = np.exp(1j * basis_phase) * basis_k1 @ Ud(np.pi / 4, basis_b, 0) @ basis_k2
        decomposer = TwoQubitBasisDecomposer(UnitaryGate(basis_unitary), basis_fidelity=0.99)

        tgt_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_phase = state.random() * 2 * np.pi
        tgt_a, tgt_b = 0.3, 0.2  # how to safely randomize?
        d1, d2, d3 = state.random(size=3) * delta
        tgt_unitary = np.exp(1j * tgt_phase) * tgt_k1 @ Ud(tgt_a + d1, tgt_b + d2, d3) @ tgt_k2
        self.check_approx_decomposition(tgt_unitary, decomposer, num_basis_uses=2)

    @combine(seed=range(10), name="seed_{seed}")
    def test_approx_supercontrolled_decompose_phase_3_use_random(self, seed, delta=0.01):
        """Approximate decomposition supercontrolled basis, random tgt (3 basis uses) (seed={seed})"""
        state = np.random.default_rng(seed)
        basis_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        basis_phase = state.random() * 2 * np.pi
        basis_b = state.random() * np.pi / 4
        basis_unitary = np.exp(1j * basis_phase) * basis_k1 @ Ud(np.pi / 4, basis_b, 0) @ basis_k2
        decomposer = TwoQubitBasisDecomposer(UnitaryGate(basis_unitary), basis_fidelity=0.99)

        tgt_k1 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_k2 = np.kron(random_unitary(2, seed=state).data, random_unitary(2, seed=state).data)
        tgt_phase = state.random() * 2 * np.pi
        tgt_a, tgt_b, tgt_c = 0.5, 0.4, 0.3
        d1, d2, d3 = state.random(size=3) * delta
        tgt_unitary = (
            np.exp(1j * tgt_phase) * tgt_k1 @ Ud(tgt_a + d1, tgt_b + d2, tgt_c + d3) @ tgt_k2
        )
        self.check_approx_decomposition(tgt_unitary, decomposer, num_basis_uses=3)


@ddt
class TestTwoQubitControlledUDecompose(CheckDecompositions):
    """Test TwoQubitControlledUDecomposer() for exact decompositions and raised exceptions"""

    @combine(seed=range(10), name="seed_{seed}")
    def test_correct_unitary(self, seed):
        """Verify unitary for different gates in the decomposition"""
        unitary = random_unitary(4, seed=seed)
        for gate in [RXXGate, RYYGate, RZZGate, RZXGate, CPhaseGate, CRZGate]:
            decomposer = TwoQubitControlledUDecomposer(gate)
            circ = decomposer(unitary)
            self.assertEqual(Operator(unitary), Operator(circ))

    def test_not_rxx_equivalent(self):
        """Test that an exception is raised if the gate is not equivalent to an RXXGate"""
        gate = SwapGate
        with self.assertRaises(QiskitError) as exc:
            TwoQubitControlledUDecomposer(gate)
        self.assertIn(
            "Equivalent gate needs to take exactly 1 angle parameter.", exc.exception.message
        )


class TestDecomposeProductRaises(QiskitTestCase):
    """Check that exceptions are raised when 2q matrix is not a product of 1q unitaries"""

    def test_decompose_two_qubit_product_gate_detr_too_small(self):
        """Check that exception raised for too-small right component"""
        kl = np.eye(2)
        kr = 0.05 * np.eye(2)
        klkr = np.kron(kl, kr)
        with self.assertRaises(QiskitError) as exc:
            decompose_two_qubit_product_gate(klkr)
        self.assertIn("detR <", exc.exception.message)

    def test_decompose_two_qubit_product_gate_detl_too_small(self):
        """Check that exception raised for too-small left component"""
        kl = np.array([[1, 0], [0, 0]])
        kr = np.eye(2)
        klkr = np.kron(kl, kr)
        with self.assertRaises(QiskitError) as exc:
            decompose_two_qubit_product_gate(klkr)
        self.assertIn("detL <", exc.exception.message)

    def test_decompose_two_qubit_product_gate_not_product(self):
        """Check that exception raised for non-product unitary"""
        klkr = Ud(1.0e-6, 0, 0)
        with self.assertRaises(QiskitError) as exc:
            decompose_two_qubit_product_gate(klkr)
        self.assertIn("decomposition failed", exc.exception.message)


if __name__ == "__main__":
    unittest.main()
