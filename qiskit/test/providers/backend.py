# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base TestCase for testing backends."""

from unittest import SkipTest

from qiskit import execute
from ..base import QiskitTestCase
from ..reference_circuits import ReferenceCircuits


class BackendTestCase(QiskitTestCase):
    """Test case for backends.

    Implementers of backends are encouraged to subclass and customize this
    TestCase, as it contains a "canonical" series of tests in order to ensure
    the backend functionality matches the specifications.

    Members:
        backend_cls (BaseBackend): backend to be used in this test case. Its
            instantiation can be further customized by overriding the
            ``_get_backend`` function.
        circuit (QuantumCircuit): circuit to be used for the tests.
    """
    backend_cls = None
    circuit = ReferenceCircuits.bell()

    def setUp(self):
        super().setUp()
        self.backend = self._get_backend()

    @classmethod
    def setUpClass(cls):
        if cls is BackendTestCase:
            raise SkipTest('Skipping base class tests')
        super().setUpClass()

    def _get_backend(self):
        """Return an instance of a Provider."""
        return self.backend_cls()  # pylint: disable=not-callable

    def test_configuration(self):
        """Test backend.configuration()."""
        configuration = self.backend.configuration()
        return configuration

    def test_properties(self):
        """Test backend.properties()."""
        properties = self.backend.properties()
        if self.backend.configuration().simulator:
            self.assertEqual(properties, None)
        return properties

    def test_status(self):
        """Test backend.status()."""
        status = self.backend.status()
        return status

    def test_run_circuit(self):
        """Test running a single circuit."""
        job = execute(self.circuit, self.backend)
        result = job.result()
        self.assertEqual(result.success, True)
        return result
