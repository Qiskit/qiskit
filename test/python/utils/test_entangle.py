# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Test of entanglement capability
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit

from qiskit.test import QiskitTestCase
from qiskit.utils import has_aer
from qiskit.utils import Entanglement

if has_aer():
    from qiskit import Aer


def get_circ_3(feature_dim, repitition) -> QuantumCircuit:
    """
    Returns:
        circ: Circuit 3 mentioned in  https://doi.org/10.1007/s42484-021-00038-w
    """

    num_par = (2 * feature_dim * repitition) + (feature_dim - 1) * repitition
    paravec = np.random.randn(num_par)
    circ = QuantumCircuit(feature_dim)
    arg_count = 0

    for _ in range(repitition):
        for i in range(circ.num_qubits):
            circ.rx(paravec[i], i)
            circ.rz(paravec[i + 1], i)
            arg_count += 2
        for i in reversed(range(circ.num_qubits - 1)):
            circ.crz(paravec[arg_count], i + 1, i)
            arg_count += 1
        circ.barrier()
    return circ


class TestEntCap(QiskitTestCase):
    """The test class"""

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_entanglement_capability(self):
        """Test entanglement capability with Aer's statevector simulator."""
        backend = Aer.get_backend("statevector_simulator")
        repitition = 1
        feature_dim = 4
        num_parameters = 100
        parametric_circuit = get_circ_3(feature_dim, repitition)

        with self.subTest("Defaults"):
            entanglement_capability = Entanglement(
                parametric_circuit, backend, num_params=num_parameters
            ).get_entanglement()
            self.assertTrue(0.0 <= entanglement_capability <= 1.0)

        with self.subTest("Include custom"):
            entanglement_capability = Entanglement(
                parametric_circuit, backend, ent_measure="von-neumann", num_params=num_parameters
            ).get_entanglement()
            self.assertTrue(0.0 <= entanglement_capability <= np.log(2))
