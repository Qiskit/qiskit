# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,broad-except

"""Backend grouped/deprecated/aliased test."""

from qiskit import register, get_backend, available_backends
from qiskit.backends.local import QasmSimulatorPy, QasmSimulatorCpp
from .common import requires_qe_access, QiskitTestCase


# Cpp backend required
try:
    cpp_backend = QasmSimulatorCpp()
except FileNotFoundError:
    _skip_cpp = True
else:
    _skip_cpp = False


class TestBackendNames(QiskitTestCase):
    """
    Test grouped/deprecated/aliased names from providers.
    """

    def test_local_groups(self):
        """test local group names are resolved correctly"""
        group_name = "local_qasm_simulator"
        backend = get_backend(group_name)
        if not _skip_cpp:
            self.assertIsInstance(backend, QasmSimulatorCpp)
        else:
            self.assertIsInstance(backend, QasmSimulatorPy)

    def test_local_deprecated(self):
        """test deprecated local backends are resolved correctly"""
        old_name = "local_qiskit_simulator"
        if not _skip_cpp:
            new_backend = get_backend(old_name)
            self.assertIsInstance(new_backend, QasmSimulatorCpp)

    @requires_qe_access
    def test_ibmq_deprecated(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """test deprecated ibmq backends are resolved correctly"""
        register(QE_TOKEN, QE_URL)
        old_name_1 = "ibmqx_qasm_simulator"
        old_name_2 = "ibmqx_hpc_qasm_simulator"
        new_name = "ibmq_qasm_simulator"
        self.assertEqual(get_backend(old_name_1), get_backend(new_name))
        self.assertEqual(get_backend(old_name_2), get_backend(new_name))

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

    def test_compact_flag(self):
        """test the compact flag for available_backends works"""
        compact_names = available_backends()
        expanded_names = available_backends(compact=False)
        self.assertIn('local_qasm_simulator', compact_names)
        self.assertIn('local_statevector_simulator', compact_names)
        self.assertIn('local_unitary_simulator', compact_names)
        self.assertIn('local_qasm_simulator_py', expanded_names)
        self.assertIn('local_statevector_simulator_py', expanded_names)
