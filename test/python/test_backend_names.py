# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,broad-except

"""Backend grouped/deprecated/aliased test."""

from qiskit import available_backends, Aer
from qiskit.backends.local import QasmSimulatorPy, QasmSimulatorCpp
from .common import (QiskitTestCase,
                     is_cpp_simulator_available,
                     requires_cpp_simulator)


class TestBackendNames(QiskitTestCase):
    """
    Test grouped/deprecated/aliased names from providers.
    """

    def test_local_groups(self):
        """test local group names are resolved correctly"""
        group_name = 'local_qasm_simulator'
        backend = Aer.get_backend(group_name)
        if is_cpp_simulator_available():
            self.assertIsInstance(backend, QasmSimulatorCpp)
        else:
            self.assertIsInstance(backend, QasmSimulatorPy)

    @requires_cpp_simulator
    def test_local_deprecated(self):
        """test deprecated local backends are resolved correctly"""
        old_name = 'local_qiskit_simulator'
        new_backend = Aer.get_backend(old_name)
        self.assertIsInstance(new_backend, QasmSimulatorCpp)

    def test_compact_flag(self):
        """Test the compact flag for available_backends works"""
        compact_names = available_backends()
        expanded_names = available_backends(compact=False)
        self.assertIn('local_qasm_simulator', compact_names)
        self.assertIn('local_statevector_simulator', compact_names)
        self.assertIn('local_unitary_simulator', compact_names)
        self.assertIn('local_qasm_simulator_py', expanded_names)
        self.assertIn('local_statevector_simulator_py', expanded_names)
