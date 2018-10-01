# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Tests for the wrapper functionality."""

import unittest

import qiskit.wrapper
from qiskit import QISKitError
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit.qobj import Qobj
from qiskit.wrapper import registered_providers, execute
from ._mockutils import DummyProvider
from .common import QiskitTestCase, requires_qe_access
from .common import is_cpp_simulator_available
from .test_backends import remove_backends_from_list


class TestWrapper(QiskitTestCase):
    """Wrapper test case."""
    def setUp(self):
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        self.circuit = QuantumCircuit(qr, cr)
        self.circuit.ccx(qr[0], qr[1], qr[2])
        self.circuit.measure(qr, cr)

    @requires_qe_access
    def test_wrapper_register_ok(self, qe_token, qe_url):
        """Test wrapper.register()."""
        qiskit.wrapper.register(qe_token, qe_url)
        backends = qiskit.wrapper.available_backends()
        backends = remove_backends_from_list(backends)
        self.log.info(backends)
        self.assertTrue(len(backends) > 0)

    @requires_qe_access
    def test_backends_with_filter(self, qe_token, qe_url):
        """Test wrapper.available_backends(filter=...)."""
        qiskit.wrapper.register(qe_token, qe_url)
        backends = qiskit.wrapper.available_backends({'local': False,
                                                      'simulator': True})
        self.log.info(backends)
        self.assertTrue(len(backends) > 0)

    def test_aer_backends(self):
        """Test wrapper.available_backends({'local': True})"""
        aer_backends = qiskit.wrapper.available_backends({'local': True})
        self.log.info(aer_backends)
        self.assertTrue(len(aer_backends) > 0)

    @requires_qe_access
    def test_register_twice(self, qe_token, qe_url):
        """Test double registration of the same credentials."""
        qiskit.wrapper.register(qe_token, qe_url)
        initial_providers = registered_providers()

        with self.assertWarns(Warning) as context_manager:
            qiskit.wrapper.register(qe_token, qe_url)

        self.assertEqual('Credentials are already in use.',
                         str(context_manager.warnings[-1].message))
        self.assertCountEqual(initial_providers, registered_providers())

    def test_register_bad_credentials(self):
        """Test registering a provider with bad credentials."""
        initial_providers = registered_providers()
        with self.assertRaises(QISKitError):
            qiskit.wrapper.register('FAKE_TOKEN', 'http://unknown')
        self.assertEqual(initial_providers, registered_providers())

    def test_register_third_party(self):
        """Test registering third party backend."""
        dummy_provider = qiskit.wrapper.register(provider_class=DummyProvider)
        dummy_backend = dummy_provider.get_backend('local_dummy_simulator')

        self.assertTrue(dummy_backend)

    def test_aer_execute_and_get_ran_qasm(self):
        """Check if the aer backend returns the ran qasm."""

        cpp_simulators = [
            'qasm_simulator',
            'statevector_simulator'
        ]

        python_simulators = [
            'qasm_simulator_py',
            'statevector_simulator_py',
            'unitary_simulator'
        ]

        aer_simulators = python_simulators
        if is_cpp_simulator_available():
            aer_simulators += cpp_simulators

        for backend_name in aer_simulators:
            with self.subTest(backend_name=backend_name):
                result = execute(self.circuit, 'qasm_simulator').result()
                self.assertIsNotNone(result.get_ran_qasm(self.circuit.name))

    def test_qobj_to_circuits_single(self):
        """Check that qobj_to_circuits's result matches the qobj ini."""
        backend = 'qasm_simulator'
        qobj_in = qiskit.wrapper.compile(self.circuit, backend, skip_transpiler=True)
        out_circuit = qiskit.wrapper.qobj_to_circuits(qobj_in)
        self.assertEqual(out_circuit[0].qasm(), self.circuit.qasm())

    def test_qobj_to_circuits_multiple(self):
        """Check that qobj_to_circuits's result with multiple circuits"""
        backend = 'qasm_simulator'
        qreg1 = QuantumRegister(2)
        qreg2 = QuantumRegister(3)
        creg1 = ClassicalRegister(2)
        creg2 = ClassicalRegister(2)
        circuit_b = QuantumCircuit(qreg1, qreg2, creg1, creg2)
        circuit_b.x(qreg1)
        circuit_b.h(qreg2)
        circuit_b.measure(qreg1, creg1)
        circuit_b.measure(qreg2[0], creg2[1])
        qobj = qiskit.wrapper.compile([self.circuit, circuit_b], backend, skip_transpiler=True)
        qasm_list = [x.qasm() for x in qiskit.wrapper.qobj_to_circuits(qobj)]
        self.assertEqual(qasm_list, [self.circuit.qasm(), circuit_b.qasm()])

    def test_qobj_to_circuits_with_qobj_no_qasm(self):
        """Verify that qobj_to_circuits returns None without QASM."""
        qobj = Qobj('abc123', {}, {}, {})
        self.assertIsNone(qiskit.wrapper.qobj_to_circuits(qobj))

    def test_compact_flag(self):
        """Test the compact flag for available_backends works"""
        compact_names = qiskit.wrapper.available_backends()
        expanded_names = qiskit.wrapper.available_backends(compact=False)
        self.assertIn('qasm_simulator', compact_names)
        self.assertIn('statevector_simulator', compact_names)
        self.assertIn('unitary_simulator', compact_names)
        self.assertIn('qasm_simulator_py', expanded_names)
        self.assertIn('statevector_simulator_py', expanded_names)


if __name__ == '__main__':
    unittest.main(verbosity=2)
