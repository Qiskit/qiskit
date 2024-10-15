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

"""Test Qiskit's AnnotatedOperation class."""

import ddt
import numpy as np

from qiskit.circuit import QuantumCircuit, twirl_circuit
from qiskit.circuit.library import CXGate, ECRGate, CZGate, iSwapGate
from qiskit.quantum_info import Operator
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt.ddt
class TestTwirling(QiskitTestCase):
    """Testing qiskit.circuit.twirl_circuit"""

    @ddt.data(CXGate, ECRGate, CZGate, iSwapGate)
    def test_twirl_circuit_equiv(self, gate):
        """Test the twirled circuit is equivalent."""
        qc = QuantumCircuit(2)
        qc.append(gate(), (0, 1))
        for i in range(100):
            with self.subTest(i):
                res = twirl_circuit(qc, gate, i)
                np.testing.assert_allclose(
                    Operator(qc), Operator(res), err_msg=f"gate: {gate} not equiv to\n{res}"
                )

    @ddt.data(CXGate, ECRGate, CZGate, iSwapGate)
    def test_many_twirls_equiv(self, gate):
        """Test the twirled circuits are equivalent if num_twirls>1."""
        qc = QuantumCircuit(2)
        qc.append(gate(), (0, 1))
        res = twirl_circuit(qc, gate, seed=424242, num_twirls=1000)
        for twirled_circuit in res:
            np.testing.assert_allclose(
                Operator(qc), Operator(twirled_circuit), err_msg=f"gate: {gate} not equiv to\n{res}"
            )
