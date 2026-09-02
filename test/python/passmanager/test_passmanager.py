# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Pass manager test cases."""

from test import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.passmanager import Pass, PassManager
from qiskit.transpiler import Target


class SetLayout(Pass):
    def __init__(self, target: Target):
        self.target = target

    def run(self, ir, context):
        if ir.num_qubits > self.target.num_qubits:
            raise ValueError("Not enough qubits")

        context.set("layout", list(range(ir.num_qubits)))
        return ir


class ResetComputationalQubits(Pass):
    def run(self, ir, context):
        if (layout := context.get("layout")) is None:
            raise RuntimeError("'layout' not available")

        ir.reset(layout)
        return ir


class TestPassManager(QiskitTestCase):
    """Pass manager tests."""

    def test_pass(self):
        """Test a simple pass setup."""

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.t(1)
        circuit.cx(0, 1)

        target = Target(num_qubits=10)

        pm = PassManager()
        pm.push(SetLayout(target))
        pm.push(ResetComputationalQubits())

        out, context = pm.run(circuit)

        self.assertIsInstance(out, QuantumCircuit)
        self.assertEqual(out.count_ops().get("reset", 0), 2)
        self.assertEqual(context.get("layout", []), [0, 1])
