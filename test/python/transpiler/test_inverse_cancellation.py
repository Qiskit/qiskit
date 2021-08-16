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
Testing inverse_cancellation
"""
from numpy import pi

from qiskit import QuantumCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import CXCancellation, Cancellation
from qiskit.transpiler import PassManager
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import RXGate, HGate, CXGate, PhaseGate 


class TestCancellation(QiskitTestCase): 

    def test_inverse_cancellation_h(self):
        qc = QuantumCircuit(2,2)
        qc.h(0)
        qc.h(0)
        pass_ = Cancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("h", gates_after)

    def test_inverse_cancellation_h3(self):
        qc = QuantumCircuit(2,2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        pass_ = Cancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("h", gates_after)
        self.assertEqual(gates_after["h"], 1)

    def test_inverse_cancellation_cx(self):
        qc = QuantumCircuit(2,2)
        qc.cx(0,1)
        qc.cx(0,1)
        pass_ = Cancellation([CXGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("cx", gates_after)
    
    def test_inverse_cancellation_rx1(self):
        qc = QuantumCircuit(2,2)
        qc.rx(pi/4, 0)
        qc.rx(-pi/4, 0)
        pass_ = Cancellation([(RXGate(pi/4), RXGate(-pi/4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("rx", gates_after)

    def test_inverse_cancellation_rx2(self):
        qc = QuantumCircuit(2,2)
        qc.rx(pi/4, 0)
        qc.rx(pi/4, 0)
        pass_ = Cancellation([(RXGate(pi/4), RXGate(-pi/4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("rx", gates_after)
        self.assertEqual(gates_after["rx"], 2)
    
    def test_inverse_cancellation_rx3(self):
        qc = QuantumCircuit(2,2)
        qc.rx(pi/2, 0)
        qc.rx(pi/4, 0)
        with self.assertRaises(TranspilerError):
            pass_ = Cancellation([RXGate(0.5)])

    def test_inverse_cancellation_hcx(self):
        qc = QuantumCircuit(2,2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(0)
        pass_ = Cancellation([HGate(), CXGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("cx", gates_after)
        self.assertEqual(gates_after["h"], 2)

    def test_inverse_cancellation_p(self):
        qc = QuantumCircuit(2,2)
        qc.p(pi/4, 0)
        qc.p(-pi/4, 0)
        pass_ = Cancellation([(PhaseGate(pi/4), PhaseGate(-pi/4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("p", gates_after)

    def test_inverse_cancellation_h4(self):
        qc = QuantumCircuit(2,2)
        qc.h(0)
        qc.h(1)
        qc.h(0)
        qc.h(1)
        pass_ = Cancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("h", gates_after)
