# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Test backend name resolution for functionality, via groups, deprecations and
aliases."""

from qiskit import IBMQ, Aer
from qiskit.backends.aer import QasmSimulatorPy, QasmSimulator
from .common import (QiskitTestCase,
                     is_cpp_simulator_available,
                     requires_cpp_simulator,
                     requires_qe_access)


class TestBackendNameResolution(QiskitTestCase):
    """
    Test backend name resolution algorithms.
    """

    def test_deprecated(self):
        """Test that deprecated names map the same backends as the new names.
        """
        deprecated_names = Aer.deprecated_backend_names()

        for oldname, newname in deprecated_names.items():
            if (newname == 'qasm_simulator' or
                    newname == 'statevector_simulator') and not is_cpp_simulator_available():
                continue

            with self.subTest(oldname=oldname, newname=newname):
                try:
                    resolved_newname = _get_first_available_backend(newname)
                    real_backend = Aer.get_backend(resolved_newname)
                except KeyError:
                    # The real name of the backend might not exist
                    pass
                else:
                    self.assertEqual(Aer.backends(oldname)[0], real_backend)

    @requires_qe_access
    def test_aliases(self, qe_token, qe_url):
        """Test that display names of devices map the same backends as the
        regular names."""
        IBMQ.enable_account(qe_token, qe_url)
        aliased_names = IBMQ.aliased_backend_names()

        for display_name, backend_name in aliased_names.items():
            with self.subTest(display_name=display_name,
                              backend_name=backend_name):
                try:
                    backend_by_name = IBMQ.get_backend(backend_name)
                except KeyError:
                    # The real name of the backend might not exist
                    pass
                else:
                    backend_by_display_name = IBMQ.get_backend(display_name)
                    self.assertEqual(backend_by_name, backend_by_display_name)
                    self.assertEqual(backend_by_display_name.name(), backend_name)

    def test_groups(self):
        """Test that aggregate group names map to the first available backend
        of their list of backends."""
        aer_groups = Aer.grouped_backend_names()
        for group_name, priority_list in aer_groups.items():
            with self.subTest(group_name=group_name,
                              priority_list=priority_list):
                target_backend = _get_first_available_backend(priority_list)
                if target_backend:
                    self.assertEqual(Aer.get_backend(group_name),
                                     Aer.get_backend(target_backend))

    def test_aliases_fail(self):
        """Test a failing backend lookup."""
        self.assertRaises(LookupError, Aer.get_backend, 'bad_name')


class TestAerBackendNames(QiskitTestCase):
    """
    Test grouped/deprecated/aliased names from providers.
    """
    def test_aer_groups(self):
        """test aer group names are resolved correctly"""
        group_name = 'qasm_simulator'
        backend = Aer.get_backend(group_name)
        if is_cpp_simulator_available():
            self.assertIsInstance(backend, QasmSimulator)
        else:
            self.assertIsInstance(backend, QasmSimulatorPy)

    @requires_cpp_simulator
    def test_aer_deprecated(self):
        """test deprecated aer backends are resolved correctly"""
        old_name = 'local_qiskit_simulator'
        new_backend = Aer.get_backend(old_name)
        self.assertIsInstance(new_backend, QasmSimulator)


def _get_first_available_backend(backend_names):
    """Gets the first available backend."""
    if isinstance(backend_names, str):
        backend_names = [backend_names]

    for backend_name in backend_names:
        try:
            return Aer.get_backend(backend_name).name()
        except LookupError:
            pass

    return None
