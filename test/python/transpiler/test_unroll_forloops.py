# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the unroll_forloops pass"""

import unittest

from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister, ClassicalRegister
from qiskit.transpiler import PassManager
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.optimization.unroll_forloops import UnrollForLoops


class TestUnrool(QiskitTestCase):
    """Test UnrollForLoops pass"""

    def test_range(self):
        """TODO"""
        qreg, creg = QuantumRegister(5, "q"), ClassicalRegister(2, "c")

        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, [0, 1, 2])

        circuit = QuantumCircuit(qreg, creg)
        circuit.for_loop(indexset, loop_parameter, body, [1, 2, 3], [1])

        expected = QuantumCircuit(qreg, creg)
        for index_loop in indexset:
            expected.rx(index_loop, [1, 2, 3])

        passmanager = PassManager()
        passmanager.append(UnrollForLoops())
        result = passmanager.run(circuit)

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
