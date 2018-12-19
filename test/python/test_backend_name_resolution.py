# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Test backend name resolution for functionality, via groups, deprecations and
aliases."""

from qiskit import IBMQ, BasicAer, LegacySimulators
from qiskit.providers.legacysimulators import QasmSimulator
from qiskit.providers.exceptions import QiskitBackendNotFoundError
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
        for provider in (BasicAer, LegacySimulators):
            deprecated_names = provider._deprecated_backend_names()

            for oldname, newname in deprecated_names.items():
                if (newname == 'qasm_simulator' or
                        newname == 'statevector_simulator') and not is_cpp_simulator_available():
                    continue

                with self.subTest(provider=provider, oldname=oldname, newname=newname):
                    try:
                        resolved_newname = _get_first_available_backend(provider, newname)
                        real_backend = provider.get_backend(resolved_newname)
                    except QiskitBackendNotFoundError:
                        # The real name of the backend might not exist
                        pass
                    else:
                        self.assertEqual(provider.backends(oldname)[0], real_backend)

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

    def test_deprecated_cpp_simulator_return_no_backend(self):
        """Test backends("local_qasm_simulator_cpp") does not return C++
        simulator if it is not installed"""
        name = "local_qasm_simulator_cpp"
        backends = LegacySimulators.backends(name)
        if is_cpp_simulator_available():
            self.assertEqual(len(backends), 1)
            self.assertIsInstance(backends[0] if backends else None, QasmSimulator)
        else:
            self.assertEqual(len(backends), 0)


class TestSimulatorBackendNames(QiskitTestCase):
    """
    Test deprecated names from providers.
    """
    @requires_cpp_simulator
    def test_legacy_deprecated(self):
        """test deprecated legacy simulators backends are resolved correctly"""
        old_name = 'local_qiskit_simulator'
        new_backend = LegacySimulators.get_backend(old_name)
        self.assertIsInstance(new_backend, QasmSimulator)


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
