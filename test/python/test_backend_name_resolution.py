# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Test backend name resolution for functionality groups, deprecations and
aliases."""

from qiskit import IBMQ, Aer
from .common import requires_qe_access, QiskitTestCase, is_cpp_simulator_available


class TestBackendNameResolution(QiskitTestCase):
    """
    Test backend resolution algorithms.
    """

    def test_deprecated(self):
        """Test that deprecated names map the same backends as the new names.
        """
        deprecated_names = Aer.deprecated_backend_names()

        for oldname, newname in deprecated_names.items():
            if newname == 'local_qasm_simulator_cpp' and not is_cpp_simulator_available():
                continue

            with self.subTest(oldname=oldname, newname=newname):
                try:
                    real_backend = Aer.backend(newname)
                except KeyError:
                    # The real name of the backend might not exist
                    pass
                else:
                    self.assertEqual(Aer.backends(oldname)[0], real_backend)

    @requires_qe_access
    def test_aliases(self, qe_token, qe_url):
        """Test that display names of devices map the same backends as the
        regular names."""
        IBMQ.use_account(qe_token, qe_url)
        aliased_names = IBMQ.aliased_backend_names()

        for display_name, backend_name in aliased_names.items():
            with self.subTest(display_name=display_name,
                              backend_name=backend_name):
                try:
                    backend_by_name = IBMQ.backend(backend_name)
                except IndexError:
                    # The real name of the backend might not exist
                    pass
                else:
                    backend_by_display_name = IBMQ.backend(display_name)
                    self.assertEqual(backend_by_name, backend_by_display_name)
                    self.assertEqual(backend_by_display_name.name(), backend_name)

    def test_aggregate(self):
        """Test that aggregate group names maps the first available backend
        of their list of backends."""
        aggregate_backends = Aer.grouped_backend_names()
        for group_name, priority_list in aggregate_backends.items():
            with self.subTest(group_name=group_name,
                              priority_list=priority_list):
                target_backend = _get_first_available_backend(priority_list)
                if target_backend:
                    self.assertEqual(Aer.backends(group_name),
                                     Aer.backends(target_backend))

    def test_aliases_fail(self):
        """Test a failing backend lookup."""
        self.assertRaises(LookupError, Aer.backend, 'bad_name')


def _get_first_available_backend(backends):
    """Gets the first available backend."""
    for backend_name in backends:
        try:
            return Aer.backend(backend_name).name()
        except LookupError:
            pass

    return None
