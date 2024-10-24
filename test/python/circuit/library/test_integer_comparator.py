# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of integer comparison circuits."""

import unittest
import numpy as np
from ddt import ddt, data, unpack

from qiskit import transpile, QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import IntegerComparator, IntegerComparatorGate
from qiskit.quantum_info import Statevector
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestIntegerComparator(QiskitTestCase):
    """Test the integer comparator circuit."""

    def assertComparisonIsCorrect(self, comp, num_state_qubits, value, geq):
        """Assert that the comparator output is correct."""
        qc = QuantumCircuit(2 * num_state_qubits)  # initialize circuit
        qc.h(list(range(num_state_qubits)))  # set equal superposition state
        qc.append(comp, list(range(comp.num_qubits)))  # add comparator

        # run simulation
        tqc = transpile(qc)  # trigger the HLS if necessary
        statevector = Statevector(tqc)
        for i, amplitude in enumerate(statevector):
            prob = np.abs(amplitude) ** 2
            if prob > 1e-6:
                # equal superposition
                self.assertEqual(True, np.isclose(1.0, prob * 2.0**num_state_qubits))
                b_value = f"{i:b}".rjust(qc.width(), "0")
                x = int(b_value[(-num_state_qubits):], 2)
                comp_result = int(b_value[-num_state_qubits - 1], 2)
                if geq:
                    self.assertEqual(x >= value, comp_result == 1)
                else:
                    self.assertEqual(x < value, comp_result == 1)

    @data(
        [1, 0, True],
        [1, 1, True],
        [2, -1, True],
        [3, 5, True],
        [3, 2, True],
        [3, 2, False],
        [4, 6, False],
    )
    @unpack
    def test_fixed_value_comparator(self, num_state_qubits, value, geq):
        """Test the fixed value comparator circuit."""
        # build the circuit with the comparator
        for use_gate in [True, False]:
            with self.subTest(use_gate=use_gate):
                constructor = IntegerComparatorGate if use_gate else IntegerComparator

                comp = constructor(num_state_qubits, value, geq=geq)
                self.assertComparisonIsCorrect(comp, num_state_qubits, value, geq)

    def test_mutability(self):
        """Test changing the arguments of the comparator."""

        comp = IntegerComparator()

        with self.subTest(msg="missing num state qubits and value"):
            with self.assertRaises(AttributeError):
                _ = str(comp.draw())

        comp.num_state_qubits = 2

        with self.subTest(msg="missing value"):
            with self.assertRaises(AttributeError):
                _ = str(comp.draw())

        comp.value = 0
        comp.geq = True

        with self.subTest(msg="updating num state qubits"):
            comp.num_state_qubits = 1
            self.assertComparisonIsCorrect(comp, 1, 0, True)

        with self.subTest(msg="updating the value"):
            comp.num_state_qubits = 3
            comp.value = 2
            self.assertComparisonIsCorrect(comp, 3, 2, True)

        with self.subTest(msg="updating geq"):
            comp.geq = False
            self.assertComparisonIsCorrect(comp, 3, 2, False)

    def test_plugin_warning(self):
        """Test the plugin for IntegerComparatorGate warns if there are insufficient aux qubits."""

        gate = IntegerComparatorGate(2, 3)
        circuit = QuantumCircuit(3)
        circuit.append(gate, circuit.qubits)

        with self.assertRaisesRegex(
            QiskitError,
            "The IntegerComparatorGate can currently only be synthesized with num_state_qubits - 1 ",
        ):
            _ = transpile(circuit)


if __name__ == "__main__":
    unittest.main()
