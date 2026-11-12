# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for provider utilities."""

from types import SimpleNamespace
from unittest.mock import Mock

from qiskit.providers.providerutils import filter_backends
from test import QiskitTestCase


class _BackendAttributes(SimpleNamespace):
    """Backend attributes supporting the legacy membership interface."""

    def __contains__(self, key):
        return hasattr(self, key)


class TestFilterBackends(QiskitTestCase):
    """Tests for filtering provider backends."""

    def test_filter_by_configuration(self):
        """Test filtering backends by a configuration attribute."""
        simulator = Mock()
        simulator.configuration.return_value = _BackendAttributes(simulator=True)
        hardware = Mock()
        hardware.configuration.return_value = _BackendAttributes(simulator=False)

        self.assertEqual([simulator], filter_backends([simulator, hardware], simulator=True))
