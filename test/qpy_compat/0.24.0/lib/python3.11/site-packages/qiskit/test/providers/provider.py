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

"""Base TestCase for testing Providers."""

from unittest import SkipTest

from ..base import QiskitTestCase


class ProviderTestCase(QiskitTestCase):
    """Test case for Providers.

    Implementers of providers are encouraged to subclass and customize this
    TestCase, as it contains a "canonical" series of tests in order to ensure
    the provider functionality matches the specifications.

    Members:
        provider_cls (BaseProvider): provider to be used in this test case. Its
            instantiation can be further customized by overriding the
            ``_get_provider`` function.
        backend_name (str): name of a backend provided by the provider.
    """

    provider_cls = None
    backend_name = ""

    def setUp(self):
        super().setUp()
        self.provider = self._get_provider()

    @classmethod
    def setUpClass(cls):
        if cls is ProviderTestCase:
            raise SkipTest("Skipping base class tests")
        super().setUpClass()

    def _get_provider(self):
        """Return an instance of a Provider."""
        return self.provider_cls()  # pylint: disable=not-callable

    def test_backends(self):
        """Test the provider has backends."""
        backends = self.provider.backends()
        self.assertTrue(len(backends) > 0)

    def test_get_backend(self):
        """Test getting a backend from the provider."""
        backend = self.provider.get_backend(name=self.backend_name)
        self.assertEqual(backend.name(), self.backend_name)
