# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,broad-except

"""Backend Name Alias Test."""

from qiskit import register, get_backend
from .common import requires_qe_access, QiskitTestCase


class TestAliases(QiskitTestCase):
    """
    Test aliased_backend_names from providers.
    """

    @requires_qe_access
    def test_ibmq_aliases(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test display names of devices as aliases for backend name."""
        register(QE_TOKEN, QE_URL)
        display_names = ['yorktown', 'tenerife', 'rueschlikon']
        if hub and group and project:
            display_names.append('austin')

        for display_name in display_names:
            backend_status = get_backend(display_name).status()
            self.assertIn('name', backend_status)
            backend_status = get_backend(display_name.capitalize()).status()
            self.assertIn('name', backend_status)

    def test_ibmq_aliases_fail(self):
        self.assertRaises(LookupError, get_backend, 'backend_name')
