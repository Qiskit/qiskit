# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Test circuits with variable parameters."""

import sympy
import qiskit
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate
from qiskit.transpiler import transpile
from qiskit.compiler import assemble_circuits
from qiskit import QiskitError
from qiskit.test import QiskitTestCase


class TestVariableParameters(QiskitTestCase):
    """QuantumCircuit Operations tests."""

    def test_gate(self):
        """Test instantiating gate with variable parmeters"""
        theta = sympy.Symbol('θ')
        theta_gate = Gate('test', 1, params=[theta])
        self.assertEqual(theta_gate.name, 'test')
        self.assertIsInstance(theta_gate.params[0], sympy.Symbol)

    def test_compile_quantum_circuit(self):
        """Test instantiating gate with variable parmeters"""
        theta = sympy.Symbol('θ')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        backend = BasicAer.get_backend('qasm_simulator')
        qc_aer = transpile(qc, backend)
        qobj = assemble_circuits(qc_aer)
        self.assertIn(theta, qobj.experiments[0].instructions[0].params)

    def test_get_variables(self):
        """Test instantiating gate with variable parmeters"""
        from qiskit.extensions.standard.rx import RXGate
        theta = sympy.Symbol('θ')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        rx = RXGate(theta)
        qc.append(rx, [qr[0]], [])
        vparams = qc.variable_table
        self.assertIs(theta, next(iter(vparams)))
        self.assertIs(rx, next(iter(next(iter(vparams[theta])))))

    def test_fix_variable(self):
        """Test setting a varaible to a constant value"""
        from qiskit.extensions.standard.rx import RXGate
        theta = sympy.Symbol('θ')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u3(0, theta, 0, qr)
        qc.variable_table[theta] = 0.5
        self.assertEqual(qc.variable_table[theta][0][0].params[0], 0.5)
        self.assertEqual(qc.variable_table[theta][1][0].params[1], 0.5)
        qc.variable_table[theta] = 0.6
        self.assertEqual(qc.variable_table[theta][0][0].params[0], 0.6)
        self.assertEqual(qc.variable_table[theta][1][0].params[1], 0.6)

    def test_multiple_variables(self):
        """Test setting a varaible to a constant value"""
        from qiskit.extensions.standard.rx import RXGate
        theta = sympy.Symbol('θ')
        x = sympy.Symbol('x')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u3(0, theta, x, qr)
        self.assertEqual(qc.variables, {theta, x})
