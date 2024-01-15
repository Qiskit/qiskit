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

"""BasicProvider Backends Test."""

from qiskit.providers.basic_provider.basic_provider import BasicProvider
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.test import QiskitTestCase


class TestBasicProviderBackends(QiskitTestCase):
    """Qiskit BasicAer Backends (Object) Tests."""

    def setUp(self):
        super().setUp()
        self.provider = BasicProvider()
        self.backend_name = "basic_simulator"

    def test_backends(self):
        """Test the provider has backends."""
        backends = self.provider.backends()
        self.assertTrue(len(backends) > 0)

    def test_get_backend(self):
        """Test getting a backend from the provider."""
        backend = self.provider.get_backend(name=self.backend_name)
        self.assertEqual(backend.name(), self.backend_name)

    def test_deprecated(self):
        """Test that deprecated names map the same backends as the new names."""

        def _get_first_available_backend(provider, backend_names):
            """Gets the first available backend."""
            if isinstance(backend_names, str):
                backend_names = [backend_names]
            for backend_name in backend_names:
                try:
                    return provider.get_backend(backend_name).name
                except QiskitBackendNotFoundError:
                    pass
            return None

        deprecated_names = BasicProvider()._deprecated_backend_names()
        for oldname, newname in deprecated_names.items():
            expected = (
                "WARNING:qiskit.providers.providerutils:Backend '%s' is deprecated. "
                "Use '%s'." % (oldname, newname)
            )
            with self.subTest(oldname=oldname, newname=newname):
                with self.assertLogs("qiskit.providers.providerutils", level="WARNING") as context:
                    resolved_newname = _get_first_available_backend(BasicProvider(), newname)
                    real_backend = BasicProvider().get_backend(resolved_newname)
                    self.assertEqual(BasicProvider().backends(oldname)[0].name, real_backend.name)
                self.assertEqual(context.output, [expected])

    def test_aliases_fail(self):
        """Test a failing backend lookup."""
        self.assertRaises(QiskitBackendNotFoundError, BasicProvider().get_backend, "bad_name")

    def test_aliases_return_empty_list(self):
        """Test backends() return an empty list if name is unknown."""
        self.assertEqual(BasicProvider().backends("bad_name"), [])
