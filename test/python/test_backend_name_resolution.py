# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test backend name resolution for functionality, via groups, deprecations and
aliases."""

from qiskit import IBMQ, BasicAer
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.test import QiskitTestCase, requires_qe_access


class TestBackendNameResolution(QiskitTestCase):
    """
    Test backend name resolution algorithms.
    """

    def test_deprecated(self):
        """Test that deprecated names map the same backends as the new names.
        """
        deprecated_names = BasicAer._deprecated_backend_names()

        for oldname, newname in deprecated_names.items():
            with self.subTest(oldname=oldname, newname=newname):
                try:
                    resolved_newname = _get_first_available_backend(BasicAer, newname)
                    real_backend = BasicAer.get_backend(resolved_newname)
                except QiskitBackendNotFoundError:
                    # The real name of the backend might not exist
                    pass
                else:
                    self.assertEqual(BasicAer.backends(oldname)[0], real_backend)

    @requires_qe_access
    def test_aliases(self, qe_token, qe_url):
        """Test that display names of devices map the same backends as the
        regular names."""
        IBMQ.enable_account(qe_token, qe_url)
        aliased_names = IBMQ._aliased_backend_names()

        for display_name, backend_name in aliased_names.items():
            with self.subTest(display_name=display_name,
                              backend_name=backend_name):
                try:
                    backend_by_name = IBMQ.get_backend(backend_name)
                except QiskitBackendNotFoundError:
                    # The real name of the backend might not exist
                    pass
                else:
                    backend_by_display_name = IBMQ.get_backend(display_name)
                    self.assertEqual(backend_by_name, backend_by_display_name)
                    self.assertEqual(backend_by_display_name.name(), backend_name)

    def test_aliases_fail(self):
        """Test a failing backend lookup."""
        self.assertRaises(QiskitBackendNotFoundError, BasicAer.get_backend, 'bad_name')

    def test_aliases_return_empty_list(self):
        """Test backends() return an empty list if name is unknown."""
        self.assertEqual(BasicAer.backends("bad_name"), [])


def _get_first_available_backend(provider, backend_names):
    """Gets the first available backend."""
    if isinstance(backend_names, str):
        backend_names = [backend_names]

    for backend_name in backend_names:
        try:
            return provider.get_backend(backend_name).name()
        except QiskitBackendNotFoundError:
            pass

    return None
