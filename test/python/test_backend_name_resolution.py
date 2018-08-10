# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Test backend name resolution for functionality groups, deprecations and
aliases."""

from qiskit import register, get_backend
from qiskit.wrapper._wrapper import _DEFAULT_PROVIDER
from .common import requires_qe_access, QiskitTestCase, is_cpp_simulator_available


class TestBackendNameResolution(QiskitTestCase):
    """
    Test backend resolution algorithms.
    """

    @requires_qe_access
    def test_deprecated(self, QE_TOKEN, QE_URL):
        """Test that deprecated names map the same backends as the new names.
        """
        register(QE_TOKEN, QE_URL)
        deprecated_names = _DEFAULT_PROVIDER.deprecated_backend_names()
        for oldname, newname in deprecated_names.items():
            if newname == 'local_qasm_simulator_cpp' and not is_cpp_simulator_available():
                continue

            with self.subTest(oldname=oldname, newname=newname):
                self.assertEqual(get_backend(oldname), get_backend(newname))

    @requires_qe_access
    # pylint: disable=unused-argument
    def test_aliases(self, QE_TOKEN, QE_URL):
        """Test that display names of devices map the same backends as the
        regular names."""
        register(QE_TOKEN, QE_URL)
        aliased_names = _DEFAULT_PROVIDER.aliased_backend_names()
        for display_name, backend_name in aliased_names.items():
            with self.subTest(display_name=display_name,
                              backend_name=backend_name):
                backend_by_name = get_backend(backend_name)
                backend_by_display_name = get_backend(display_name)
                self.assertEqual(backend_by_name, backend_by_display_name)
                self.assertEqual(backend_by_display_name['name'], backend_name)

    def test_aggregate(self):
        """Test that aggregate group names maps the first available backend
        of their list of backends."""
        aggregate_backends = _DEFAULT_PROVIDER.grouped_backend_names()
        for group_name, priority_list in aggregate_backends.items():
            with self.subTest(group_name=group_name,
                              priority_list=priority_list):
                target_backend = _get_first_available_backend(priority_list)
                if target_backend:
                    self.assertEqual(get_backend(group_name),
                                     get_backend(target_backend))

    def test_aliases_fail(self):
        """Test a failing backend lookup."""
        self.assertRaises(LookupError, get_backend, 'bad_name')


def _get_first_available_backend(backends):
    """Gets the first available backend."""
    for backend_name in backends:
        try:
            get_backend(backend_name)
            return backend_name
        except LookupError:
            pass

    return None
