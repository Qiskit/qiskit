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

"""Test gate reduction by axis angle analysis"""

import math
import numpy as np
from qiskit.test import QiskitTestCase

from qiskit import QuantumCircuit
from qiskit.extensions.unitary import UnitaryGate
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes.optimization.axis_angle_analysis import _su2_axis_angle
from qiskit.transpiler.passes.optimization.axis_angle_analysis import AxisAngleAnalysis
from qiskit.transpiler.passes.optimization.axis_angle_reduction import AxisAngleReduction


class TestAxisAngleReduction(QiskitTestCase):

    """Test the AxisAngleReduction pass."""

    def setUp(self):
        super().setUp()
        self.pma = PassManager(AxisAngleAnalysis())
        self.pmr = PassManager(AxisAngleReduction())

    def test_axis_angle(self):
        """test axis angle calc"""
        from qiskit.circuit.library.standard_gates import (U1Gate, TGate,
                                                           RXGate, XGate,
                                                           IGate, RZGate, HGate)

        xgate = XGate()
        tgate = TGate()
        u1 = U1Gate(np.pi/3)
        rz = RZGate(np.pi/3)
        rx = RXGate(np.pi/2)
        hgate = HGate()
        iden = IGate()
        self.assertTrue(axis_angle_phase_equal(_su2_axis_angle(xgate.to_matrix()),
                                               (np.array([1, 0, 0]), np.pi, np.pi/2)))
        self.assertTrue(axis_angle_phase_equal(_su2_axis_angle(tgate.to_matrix()),
                                               (np.array([0, 0, 1]), np.pi/4, np.pi/8)))
        self.assertTrue(axis_angle_phase_equal(_su2_axis_angle(u1.to_matrix()),
                                               (np.array([0, 0, 1]), np.pi/3, np.pi/6)))
        self.assertTrue(axis_angle_phase_equal(_su2_axis_angle(rz.to_matrix()),
                                               (np.array([0, 0, 1]), np.pi/3, 0)))
        self.assertTrue(axis_angle_phase_equal(_su2_axis_angle(rx.to_matrix()),
                                               (np.array([1, 0, 0]), np.pi/2, 0)))
        self.assertTrue(axis_angle_phase_equal(_su2_axis_angle(hgate.to_matrix()),
                                               (np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2]),
                                                np.pi, np.pi/2)))
        self.assertTrue(axis_angle_phase_equal(_su2_axis_angle(iden.to_matrix()),
                                               (np.array([0, 0, 0]), 0, 0)))

    def test_axis_angle_analysis(self):
        """test axis angle analysis"""
        altp = QuantumCircuit(1, global_phase=np.pi, name='altp')
        altp.rz(np.pi/2, 0)
        altpgate = altp.to_gate()
        circ = QuantumCircuit(1)
        circ.rz(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        circ.append(altpgate, [0])
        circ.rx(np.pi/2, 0)
        circ.x(0)
        self.pma.run(circ)
        angles = np.array([np.pi/2, np.pi/2, 3*np.pi/2, np.pi/2, np.pi])
        sym_order = np.array([4, 4, 4, 4, 2])
        axes = [(0, 0, 1),
                (0, 0, 1),
                (0, 0, -1),
                (1, 0, 0),
                (1, 0, 0)]
        dfprop = self.pma.property_set['axis-angle']
        self.assertTrue(np.allclose(dfprop.angle, angles))
        self.assertTrue(np.allclose(dfprop.symmetry_order, sym_order))
        for axis1, axis2 in zip(dfprop.axis, axes):
            self.assertTrue(np.allclose(axis1, axis2))

    def test_2q_noninteracting(self):
        """test two qubit set of non-interacting operations (except 1 cx)"""
        altp = QuantumCircuit(1, global_phase=np.pi, name='altp')
        #altp = QuantumCircuit(1, global_phase=0, name='altp')
        altp.rz(np.pi/2, 0)
        altpgate = altp.to_gate()
        circ = QuantumCircuit(2)
        # z-axis, qubit 0
        circ.rz(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        circ.append(altpgate, [0])  # test custom gate
        # x-axis, qubit 0
        circ.rx(np.pi/2, 0)
        circ.x(0)
        # zx-axis, qubit 0
        circ.h(0)
        circ.h(0)
        circ.h(0)
        # -zx-axis, qubit 0
        circ.rv(-1, 0, -1, 0)
        circ.rv(-2.5, 0, -2.5, 0)
        # z-axis, qubit 1
        circ.rz(np.pi/2, 1)
        circ.t(1)
        # y-axis, qubit 1
        circ.ry(np.pi/3, 1)
        circ.y(1)
        circ.r(np.pi/3, np.pi/2, 1)
        # cos(pi/3)x + sin(pi/3)y axis, qubit 1, 2 parameters
        circ.r(np.pi/2, np.pi/3, 1)
        circ.r(np.pi/3, np.pi/3, 1)
        circ.cx(0, 1)  # check interruption by cx
        circ.r(np.pi/3, np.pi/3, 1)
        passmanager = PassManager()
        passmanager.append(AxisAngleReduction())
        ccirc = passmanager.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_symmetric_cancellation(self):
        """Test symmetry-based cancellation works."""
        circ = QuantumCircuit(2)
        circ.z(0)
        circ.z(0)
        circ.z(0)
        circ.s(0)
        circ.s(0)
        circ.s(0)
        circ.s(0)
        circ.z(0)
        passmanager = PassManager()
        passmanager.append(AxisAngleReduction())
        ccirc = passmanager.run(circ)
        self.assertEqual(len(ccirc), 0)

    def test_unitary_gate(self):
        """Test unitary gate doesn't cause issues in params."""
        circ = QuantumCircuit(1)
        uxgate = UnitaryGate([[0, 1], [1, 0]])
        uzgate = UnitaryGate([[1, 0], [0, -1]])
        circ.x(0)
        circ.append(uxgate, [0])
        circ.append(uzgate, [0])
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(1)
        expected.z(0)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(Operator(ccirc), Operator(expected))

    def test_non_gate(self):
        """Test non-gate (barrier)."""
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.x(0)
        circ.barrier(0)
        circ.x(0)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(1)
        expected.barrier(0)
        expected.x(0)
        self.assertEqual(ccirc, expected)

    def test_global_phase_01(self):
        """Test no specified basis, rz"""
        circ = QuantumCircuit(1)
        circ.rz(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        ccirc = self.pmr.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_global_phase_02(self):
        """Test no specified basis, p"""
        circ = QuantumCircuit(1)
        circ.p(np.pi/2, 0)
        circ.rz(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        ccirc = self.pmr.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_global_phase_custom_gate_first(self):
        """Test custom gate applied first.

        Currently, pass prioritizes first variable gate as replacement gate."""
        altp = QuantumCircuit(1, global_phase=np.pi/3)
        altp.p(np.pi/3, [0])
        mygate = altp.to_gate()
        circ = QuantumCircuit(1)
        circ.append(mygate, [0])
        circ.p(np.pi/2, 0)
        circ.rz(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        ccirc = self.pmr.run(circ)
        self.assertEqual(Operator(circ), (Operator(ccirc)))

    def test_global_phase_custom_gate_second(self):
        """Test custom gate applied not first."""
        altp = QuantumCircuit(1, global_phase=np.pi/3)
        altp.p(np.pi/3, [0])
        mygate = altp.to_gate()
        circ = QuantumCircuit(1)
        circ.rz(np.pi/2, 0)
        circ.append(mygate, [0])
        circ.p(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        ccirc = self.pmr.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_global_phase_symmetry(self):
        """Test global phase for symmetry cancellation"""
        altp = QuantumCircuit(1, global_phase=np.pi/3)
        mygate = altp.to_gate()
        circ = QuantumCircuit(1)
        circ.rz(np.pi/3, 0)
        circ.append(mygate, [0])
        circ.p(np.pi/3, 0)
        circ.p(np.pi/3, 0)
        ccirc = self.pmr.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_global_phase_cancellation(self):
        """Test global phase for complete symmetry cancellation"""
        circ = QuantumCircuit(1, global_phase=np.pi/2)
        circ.rz(np.pi/2, 0)
        circ.rz(np.pi/2, 0)
        circ.rz(np.pi/2, 0)
        circ.rz(np.pi/2, 0)
        ccirc = self.pmr.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(len(ccirc), 0)

    def test_no_1q_gates(self):
        """test no single qubit gates in circuit"""
        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.cx(0, 1)
        ccirc = self.pmr.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_cx_cancellation(self):
        """test no single qubit gates in circuit"""
        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.cx(0, 1)
        circ.x(1)
        ccirc = self.pmr.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(len(ccirc), 1)

    def test_crx_reduction(self):
        """test no single qubit gates in circuit"""
        circ = QuantumCircuit(2)
        circ.crx(np.pi/2, 0, 1)
        circ.crx(np.pi/2, 0, 1)
        circ.x(1)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(2)
        expected.crx(np.pi, 0, 1)
        expected.x(1)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_crx_cancellation(self):
        """test crx gate cancellation in presence of x-gate"""
        circ = QuantumCircuit(2)
        circ.crx(np.pi, 0, 1)
        circ.crx(np.pi, 0, 1)
        circ.x(1)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(2)
        #expected.crx(2*np.pi, 0, 1)
        expected.p(np.pi, 0)
        expected.x(1)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_crx_src_block(self):
        """test block merge on source qubit"""
        circ = QuantumCircuit(2)
        circ.crx(np.pi, 0, 1)
        circ.x(0)
        circ.crx(np.pi, 0, 1)
        circ.x(0)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(2)
        expected.crx(np.pi, 0, 1)
        expected.x(0)
        expected.crx(np.pi, 0, 1)
        expected.x(0)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_repeat_until_complete(self):
        """repeat reduction until can't reduce anymore"""
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.y(0)
        circ.y(0)
        circ.x(0)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(1)
        self.assertEqual(ccirc, expected)

    def test_crx_tgt_no_block(self):
        """test do not block merge on target qubit of same axis"""
        circ = QuantumCircuit(2)
        circ.crx(np.pi, 0, 1)
        circ.x(1)
        circ.crx(np.pi, 0, 1)
        circ.x(1)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(2)
        expected.p(np.pi, 0)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_crx_tgt_block(self):
        """test block merge on target qubit of same axis"""
        circ = QuantumCircuit(2)
        circ.crx(np.pi, 0, 1)
        circ.z(1)
        circ.crx(np.pi, 0, 1)
        circ.x(1)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(2)
        expected.crx(np.pi, 0, 1)
        expected.z(1)
        expected.crx(np.pi, 0, 1)
        expected.x(1)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_dual_rx(self):
        """test merging of two rx gates"""
        circ = QuantumCircuit(1)
        circ.rx(np.pi, 0)
        circ.rx(np.pi, 0)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(1)
        expected.rx(2 * np.pi, 0)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_dual_crx(self):
        """test merging two crx gates"""
        circ = QuantumCircuit(2)
        circ.crx(np.pi, 0, 1)
        circ.crx(np.pi, 0, 1)
        expected = QuantumCircuit(2)
        expected.p(np.pi, 0)
        ccirc = self.pmr.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_4crx(self):
        """test complete cancellation of 4 crx gates"""
        circ = QuantumCircuit(2)
        circ.crx(np.pi, 0, 1)
        circ.crx(np.pi, 0, 1)
        circ.crx(np.pi, 0, 1)
        circ.crx(np.pi, 0, 1)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(2)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_6crx(self):
        """test reduction of 6 crx gates"""
        circ = QuantumCircuit(2)
        circ.crx(np.pi, 0, 1)
        circ.crx(np.pi, 0, 1)
        circ.crx(np.pi, 0, 1)
        circ.crx(np.pi, 0, 1)
        circ.crx(np.pi, 0, 1)
        circ.crx(np.pi, 0, 1)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(2)
        expected.p(np.pi, 0)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_zero_angle_sandwich(self):
        """test zero angle common axis"""
        circ = QuantumCircuit(1)
        circ.rz(np.pi/4, 0)
        circ.rz(0, 0)
        circ.rz(np.pi/4, 0)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(1)
        expected.rz(np.pi/2, 0)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_zero_angle_end(self):
        """test zero angle common axis"""
        circ = QuantumCircuit(1)
        circ.rz(np.pi/4, 0)
        circ.rz(0, 0)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(1)
        expected.rz(np.pi/4, 0)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_cx_common_target(self):
        """test cx commutation with common target"""
        circ = QuantumCircuit(3)
        circ.cx(0, 1)
        circ.cx(2, 1)
        circ.cx(0, 1)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(3)
        expected.cx(2, 1)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_cx_common_target_blocking_source(self):
        """test cx commutation with block on source qubit."""
        circ = QuantumCircuit(3)
        circ.cx(0, 1)
        circ.x(0)
        circ.cx(2, 1)
        circ.cx(0, 1)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(3)
        expected.cx(0, 1)
        expected.x(0)
        expected.cx(2, 1)
        expected.cx(0, 1)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)

    def test_cx_common_target_blocking(self):
        """test cx commutation with block on target qubit"""
        circ = QuantumCircuit(3)
        circ.cx(0, 1)
        circ.h(0)
        circ.h(1)
        circ.cx(0, 1)
        ccirc = self.pmr.run(circ)
        expected = QuantumCircuit(3)
        expected.cx(0, 1)
        expected.h(0)
        expected.h(1)
        expected.cx(0, 1)
        self.assertEqual(Operator(circ), Operator(ccirc))
        self.assertEqual(ccirc, expected)


def axis_angle_phase_equal(tup1, tup2):
    """tup is 3 component tuple of (np.array, angle, phase)"""
    return all((np.allclose(tup1[0], tup2[0]),
                math.isclose(tup1[1], tup2[1]),
                math.isclose(tup1[2], tup2[2])))
