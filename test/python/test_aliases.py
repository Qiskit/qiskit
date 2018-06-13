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

    def test_local_aliases(self):
        full_names = ['local_qasm_simulator_py', 'local_statevector_simulator_py',
                      'local_unitary_simulator_py']
        short_names = ['local_qasm_simulator', 'local_statevector_simulator',
                       'local_unitary_simulator']
        for short_name, full_name in zip(short_names, full_names):
            backend_status_short = get_backend(short_name).status
            backend_status_full = get_backend(full_name).status
            self.assertIn('name', backend_status_short)
            self.assertEqual(backend_status_full['name'], full_name)

    @requires_qe_access
    def test_ibmq_aliases(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test display names of devices as aliases for backend name."""
        register(QE_TOKEN, QE_URL)
        backend_names = ['ibmqx2', 'ibmqx4', 'ibmqx5']
        display_names = ['ibmq_5_yorktown', 'ibmq_5_tenerife', 'ibmq_16_rueschlikon']
        if hub and group and project:
            display_names.append('ibmq_20_austin')
            backend_names.append('QS1_1')
        for backend_name, display_name in zip(backend_names, display_names):
            backend_status_backend = get_backend(backend_name).status
            backend_status_display = get_backend(display_name).status
            self.assertEqual(backend_status_backend['name'], backend_name)
            self.assertEqual(backend_status_display['name'], backend_name)

    def test_aliases_fail(self):
        self.assertRaises(LookupError, get_backend, 'bad_name')
