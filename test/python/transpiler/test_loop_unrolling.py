# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for loop unrolling passes"""

from ddt import ddt, data

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Clbit, Qubit, Parameter
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.optimization import UnrollForLoops, ForLoopBodyOptimizer
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.test import QiskitTestCase
from qiskit.quantum_info import Operator
from math import pi


@ddt
class TestUnrollForLoop(QiskitTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._pass = UnrollForLoops()

    @data(1, 2, 3)
    def test_only_for_loop(self, nloops):
        body = QuantumCircuit(1)
        body.rx(0.1, 0)
        circ = QuantumCircuit(1)
        circ.for_loop(range(nloops), None, body.copy(), [0], [])
        ccirc = self._pass(circ)
        self.assertEqual(Operator(body.repeat(nloops)), Operator(ccirc))

    @data(1, 2, 3)
    def test_parametrized_for_loop(self, nloops):
        body = QuantumCircuit(1)
        θ = Parameter("θ")
        body.rx(θ + pi / 2, 0)
        circ = QuantumCircuit(1)
        circ.for_loop(range(nloops), θ, body.copy(), [0], [])
        ccirc = self._pass(circ)

        expected = QuantumCircuit(1)
        for i in range(nloops):
            expected.rx(i, 0)
        self.assertEqual(Operator(expected), Operator(ccirc))

    def test_for_loop_compound(self):
        body = QuantumCircuit(1)
        body.rx(pi, 0)
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.for_loop(range(2), None, body, [0], [])
        circ.y(0)

        expected = QuantumCircuit(1)
        expected.x(0)
        expected.rx(pi, 0)
        expected.rx(pi, 0)
        expected.y(0)
        self.assertEqual(expected, self._pass(circ))

    @data(1, 2, 3)
    def test_multiple_loops(self, nloops):
        body = QuantumCircuit(1)
        θ = Parameter("θ")
        body.rx(θ + pi / 2, 0)
        circ = QuantumCircuit(1)
        circ.for_loop(range(nloops), θ, body.copy(), [0], [])
        ccirc = self._pass(circ)

        expected = QuantumCircuit(1)
        for i in range(nloops):
            expected.rx(i, 0)
        self.assertEqual(Operator(expected), Operator(ccirc))
        
@ddt
class TestForLoopBodyOptimizer(QiskitTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._pass = UnrollForLoops()

    def test_pre_opt_1(self):
        pre_opt_cnt = 1
        circ = QuantumCircuit(2)
        circ.rx(0.1, 0)
        circ.rx(0.2, 0)        
        circ.x(0)
        # define for loop
        body = QuantumCircuit(1)
        body.rx(pi/2, 0)
        circ.for_loop(range(2), None, body, [0], [])
        basis_gates = ["x", "rx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, pre_opt_cnt=pre_opt_cnt)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected.rx(0.1, 0)
        expected.rx(0.2, 0)
        expected.rx(3 * pi / 2, 0)
        expected.for_loop(range(1), None, body, [0], [])

        self.assertEqual(ccirc, expected)

    def test_pre_opt_2(self):
        pre_opt_cnt = 2        
        circ = QuantumCircuit(2)
        circ.rx(0.1, 0)
        circ.rx(0.2, 0)        
        circ.x(0)
        # define for loop
        body = QuantumCircuit(1)
        body.rx(pi/2, 0)
        circ.for_loop(range(2), None, body, [0], [])
        basis_gates = ["x", "rx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, pre_opt_cnt=pre_opt_cnt)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected.rx(0.1, 0)
        expected.rx(pi / 2 + pi + 0.2, 0)
        expected.for_loop(range(1), None, body, [0], [])

        self.assertEqual(ccirc, expected)

    def test_pre_opt_cnot(self):
        """
        Test that the circuit,
                ┌─────────┐┌───┐┌───┐┌───────────┐
           q_0: ┤ Rx(0.1) ├┤ X ├┤ X ├┤0          ├
                └─────────┘└─┬─┘└───┘│  For_loop │
           q_1: ─────────────■───────┤1          ├
                                     └───────────┘
        where the body of the for-loop is
                ┌─────────┐     
           q_0: ┤ Rx(π/2) ├──■──
                └─────────┘┌─┴─┐
           q_1: ───────────┤ X ├
                           └───┘
        becomes,
                ┌────────────┐┌───┐     ┌───────────┐
           q_0: ┤ Rx(4.8124) ├┤ X ├──■──┤0          ├
                └────────────┘└─┬─┘┌─┴─┐│  For_loop │
           q_1: ────────────────■──┤ X ├┤1          ├
                                   └───┘└───────────┘
        """
        pre_opt_cnt = 3   
        circ = QuantumCircuit(2)
        circ.rx(0.1, 0)
        circ.cx(1, 0)        
        circ.x(0)
        # define for loop
        body = QuantumCircuit(2)
        body.rx(pi/2, 0)
        body.cx(0, 1)
        circ.for_loop(range(2), None, body, [0, 1], [])
        basis_gates = ["x", "rx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, pre_opt_cnt=pre_opt_cnt)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected.rx(0.1, 0)
        expected.rx(pi / 2 + pi + 0.2, 0)
        expected.for_loop(range(1), None, body, [0, 1], [])
        breakpoint()
        self.assertEqual(ccirc, expected)

    def test_pre_opt_breadth(self):
        """
        Test that the circuit,
        """
        pre_opt_cnt = 40
        circ = QuantumCircuit(3)
        circ.rx(0.1, 0)
        circ.x(0)
        circ.cx(1, 2)
        circ.y(2)
        # define for loop
        body = QuantumCircuit(2)
        body.rx(pi/2, 0)
        body.cx(0, 1)
        circ.for_loop(range(2), None, body, [0, 1], [])
        basis_gates = ["x", "rx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, pre_opt_cnt=pre_opt_cnt)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected.rx(0.1, 0)
        expected.rx(pi / 2 + pi + 0.2, 0)
        expected.for_loop(range(1), None, body, [0, 1], [])
        breakpoint()
        self.assertEqual(ccirc, expected)
        
    def test_post_opt(self):
        circ = QuantumCircuit(2)
        # define for loop
        body = QuantumCircuit(1)
        body.rx(pi/2, 0)
        circ.for_loop(range(2), None, body, [0], [])
        circ.rx(0.1, 0)
        circ.rx(0.2, 0)        
        circ.x(0)
        basis_gates = ["x", "rx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        print('')

        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, post_opt_cnt=3)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected.for_loop(range(1), None, body, [0], [])
        expected.rx(3 * pi / 2 + 0.1 + 0.2, 0)

        print(circ)
        print(ccirc)
        print(expected)
        breakpoint()
        self.assertEqual(ccirc, expected)
