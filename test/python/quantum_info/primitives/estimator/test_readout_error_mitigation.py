# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for ReadoutErrorMitigation."""

import unittest
from test import combine

from ddt import ddt

from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.primitives import PauliEstimator
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeBogota
from qiskit.utils import has_aer
from qiskit.result.mitigation.local_readout_mitigator import LocalReadoutMitigator

if has_aer():
    from qiskit import Aer


@ddt
class TestReadoutErrorMitigation(QiskitTestCase):
    """Test ReadoutErrorMitigation"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    @combine(method=["local", "correlated", "mthree"])
    def test_readout_error_mitigation(self, method):
        """test for readout error mitigation"""
        backend = Aer.get_backend("aer_simulator").from_backend(FakeBogota())
        backend.set_options(seed_simulator=15)
        mitigator = LocalReadoutMitigator(backend=backend, qubits=[0, 1])
        with PauliEstimator(
            [self.ansatz], [self.observable], backend=backend, mitigator=mitigator
        ) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15)
            result = est([0, 1, 1, 2, 3, 5])
        self.assertAlmostEqual(result.values[0], -1.305, places=2)
