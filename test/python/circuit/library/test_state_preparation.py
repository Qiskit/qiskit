# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
StatePreparation test.
"""

import unittest
import math
import numpy as np
from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import StatePreparation


@ddt
class TestStatePreparation(QiskitTestCase):
    """Test initialization with StatePreparation class"""

    def test_prepare_from_label(self):
        """Prepare state from label."""
        desired_sv = Statevector.from_label("01+-lr")
        qc = QuantumCircuit(6)
        qc.prepare_state("01+-lr", range(6))
        actual_sv = Statevector(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_prepare_from_int(self):
        """Prepare state from int."""
        desired_sv = Statevector.from_label("110101")
        qc = QuantumCircuit(6)
        qc.prepare_state(53, range(6))
        actual_sv = Statevector(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_prepare_from_list(self):
        """Prepare state from list."""
        desired_sv = Statevector([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        qc = QuantumCircuit(2)
        qc.prepare_state([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        actual_sv = Statevector(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_nonzero_state_incorrect(self):
        """Test final state incorrect if initial state not zero"""
        desired_sv = Statevector([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.prepare_state([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        actual_sv = Statevector(qc)
        self.assertFalse(desired_sv == actual_sv)

    @data(2, "11", [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
    def test_inverse(self, state):
        """Test inverse of StatePreparation"""
        qc = QuantumCircuit(2)
        stateprep = StatePreparation(state)
        qc.append(stateprep, [0, 1])
        qc.append(stateprep.inverse(), [0, 1])
        self.assertTrue(np.allclose(Operator(qc).data, np.identity(2**qc.num_qubits)))

    def test_double_inverse(self):
        """Test twice inverse of StatePreparation"""
        desired_sv = Statevector([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        qc = QuantumCircuit(2)
        stateprep = StatePreparation([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        qc.append(stateprep.inverse().inverse(), [0, 1])
        actual_sv = Statevector(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_incompatible_state_and_qubit_args(self):
        """Test error raised if number of qubits not compatible with state arg"""
        qc = QuantumCircuit(3)
        with self.assertRaises(QiskitError):
            qc.prepare_state("11")

    def test_incompatible_int_state_and_qubit_args(self):
        """Test error raised if number of qubits not compatible with  integer state arg"""
        # pylint: disable=pointless-statement
        with self.assertRaises(QiskitError):
            stateprep = StatePreparation(5, num_qubits=2)
            stateprep.definition

    def test_int_state_and_no_qubit_args(self):
        """Test automatic determination of qubit number"""
        stateprep = StatePreparation(5)
        self.assertEqual(stateprep.num_qubits, 3)

    def test_repeats(self):
        """Test repeat function repeats correctly"""
        qc = QuantumCircuit(2)
        qc.append(StatePreparation("01").repeat(2), [0, 1])
        self.assertEqual(qc.decompose().count_ops()["state_preparation"], 2)


if __name__ == "__main__":
    unittest.main()
