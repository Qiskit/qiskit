# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for issue 13935: obscure error message when composing incompatible circuits."""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.test import QiskitTestCase

class TestIssue13935(QiskitTestCase):
    """Test for issue 13935."""

    def test_compose_more_qubits_error_message(self):
        """Test that compose raises a descriptive error when 'other' has more qubits."""
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)
        with self.assertRaisesRegex(
            CircuitError, 
            r"The circuit being composed \(2 qubits\) is larger than the destination circuit \(1 qubits\)\."
        ):
            qc1.compose(qc2)

    def test_compose_more_clbits_error_message(self):
        """Test that compose raises a descriptive error when 'other' has more clbits."""
        qc1 = QuantumCircuit(1, 1)
        qc2 = QuantumCircuit(1, 2)
        with self.assertRaisesRegex(
            CircuitError, 
            r"The circuit being composed \(2 clbits\) is larger than the destination circuit \(1 clbits\)\."
        ):
            qc1.compose(qc2)
