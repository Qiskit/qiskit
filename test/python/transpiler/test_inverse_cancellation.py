# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Testing InverseCancellation
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import InverseCancellation
from qiskit.transpiler import PassManager
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import RXGate, HGate, CXGate, PhaseGate, XGate


class TestInverseCancellation(QiskitTestCase):
    def test_InverseCancellation_h(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        pass_ = InverseCancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("h", gates_after)

    def test_InverseCancellation_h3(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        pass_ = InverseCancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("h", gates_after)
        self.assertEqual(gates_after["h"], 1)

    def test_InverseCancellation_cx(self):
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        pass_ = InverseCancellation([CXGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("cx", gates_after)

    def test_InverseCancellation_rx1(self):
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 4, 0)
        qc.rx(-np.pi / 4, 0)
        pass_ = InverseCancellation([(RXGate(np.pi / 4), RXGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("rx", gates_after)

    def test_InverseCancellation_rx2(self):
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 4, 0)
        qc.rx(np.pi / 4, 0)
        pass_ = InverseCancellation([(RXGate(np.pi / 4), RXGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("rx", gates_after)
        self.assertEqual(gates_after["rx"], 2)

    def test_InverseCancellation_rx3(self):
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi / 4, 0)
        with self.assertRaises(TranspilerError):
            pass_ = InverseCancellation([RXGate(0.5)])

    def test_InverseCancellation_hcx(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(0)
        pass_ = InverseCancellation([HGate(), CXGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("cx", gates_after)
        self.assertEqual(gates_after["h"], 2)

    def test_InverseCancellation_p(self):
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("p", gates_after)

    def test_InverseCancellation_h4(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        qc.h(0)
        qc.h(1)
        pass_ = InverseCancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("h", gates_after)

    def test_InverseCancellation_rx4(self):
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 4, 0)
        qc.rx(np.pi / 4, 0)
        with self.assertRaises(TranspilerError):
            pass_ = InverseCancellation([(RXGate(np.pi / 4))])

    def test_InverseCancellation_herror(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        with self.assertRaises(TranspilerError):
            pass_ = InverseCancellation(["h"])

    def test_InverseCancellation_hcx(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        qc.x(0)
        qc.x(0)
        qc.h(0)
        pass_ = InverseCancellation([HGate(), XGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("x", gates_after)
        self.assertEqual(gates_after["h"], 2)
        