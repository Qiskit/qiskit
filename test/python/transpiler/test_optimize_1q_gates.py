# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the optimize-1q-gate pass"""

import unittest
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, Unroller
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.circuit import Parameter
from qiskit.transpiler.exceptions import TranspilerError


class TestOptimize1qGates(QiskitTestCase):
    """Test for 1q gate optimizations. """

    def test_dont_optimize_id(self):
        """Identity gates are like 'wait' commands.
        They should never be optimized (even without barriers).

        See: https://github.com/Qiskit/qiskit-terra/issues/2373
        """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.i(qr)
        circuit.i(qr)
        dag = circuit_to_dag(circuit)

        pass_ = Optimize1qGates()
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_optimize_h_gates_pass_manager(self):
        """Transpile: qr:--[H]-[H]-[H]-- == qr:--[u2]-- """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.h(qr[0])

        expected = QuantumCircuit(qr)
        expected.u2(0, np.pi, qr[0])

        passmanager = PassManager()
        passmanager.append(Unroller(['u2']))
        passmanager.append(Optimize1qGates())
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_1q_gates_collapse_identity_equivalent(self):
        """test optimize_1q_gates removes u1(2*pi) rotations.

        See: https://github.com/Qiskit/qiskit-terra/issues/159
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[1], qr[0])
        qc.u1(2 * np.pi, qr[0])
        qc.cx(qr[1], qr[0])
        qc.u1(np.pi / 2, qr[0])  # these three should combine
        qc.u1(np.pi, qr[0])  # to identity then
        qc.u1(np.pi / 2, qr[0])  # optimized away.
        qc.cx(qr[1], qr[0])
        qc.u1(np.pi, qr[1])
        qc.u1(np.pi, qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        dag = circuit_to_dag(qc)
        simplified_dag = Optimize1qGates().run(dag)

        num_u1_gates_remaining = len(simplified_dag.named_nodes('u1'))
        self.assertEqual(num_u1_gates_remaining, 0)

    def test_ignores_conditional_rotations(self):
        """Conditional rotations should not be considered in the chain.

        qr0:--[U1]-[U1]-[U1]-[U1]-    qr0:--[U1]-[U1]-
               ||   ||                       ||   ||
        cr0:===.================== == cr0:===.====.===
                    ||                            ||
        cr1:========.=============    cr1:========.===
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.u1(0.1, qr).c_if(cr, 1)
        circuit.u1(0.2, qr).c_if(cr, 3)
        circuit.u1(0.3, qr)
        circuit.u1(0.4, qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.u1(0.1, qr).c_if(cr, 1)
        expected.u1(0.2, qr).c_if(cr, 3)
        expected.u1(0.7, qr)

        pass_ = Optimize1qGates()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_in_the_back(self):
        """Optimizations can be in the back of the circuit.
        See https://github.com/Qiskit/qiskit-terra/issues/2004.

        qr0:--[U1]-[U1]-[H]--    qr0:--[U1]-[H]--
        """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u1(0.3, qr)
        circuit.u1(0.4, qr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.u1(0.7, qr)
        expected.h(qr)

        pass_ = Optimize1qGates()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_single_parameterized_circuit(self):
        """Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter('theta')

        qc.u1(0.3, qr)
        qc.u1(0.4, qr)
        qc.u1(theta, qr)
        qc.u1(0.1, qr)
        qc.u1(0.2, qr)
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(qr)
        expected.u1(0.7, qr)
        expected.u1(theta, qr)
        expected.u1(0.3, qr)

        after = Optimize1qGates().run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_parameterized_circuits(self):
        """Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter('theta')

        qc.u1(0.3, qr)
        qc.u1(0.4, qr)
        qc.u1(theta, qr)
        qc.u1(0.1, qr)
        qc.u1(0.2, qr)
        qc.u1(theta, qr)
        qc.u1(0.3, qr)
        qc.u1(0.2, qr)

        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(qr)
        expected.u1(0.7, qr)
        expected.u1(theta, qr)
        expected.u1(0.3, qr)
        expected.u1(theta, qr)
        expected.u1(0.5, qr)

        after = Optimize1qGates().run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_parameterized_expressions_in_circuits(self):
        """Expressions of Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter('theta')
        phi = Parameter('phi')

        sum_ = theta + phi
        product_ = theta * phi
        qc.u1(0.3, qr)
        qc.u1(0.4, qr)
        qc.u1(theta, qr)
        qc.u1(phi, qr)
        qc.u1(sum_, qr)
        qc.u1(product_, qr)
        qc.u1(0.3, qr)
        qc.u1(0.2, qr)

        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(qr)
        expected.u1(0.7, qr)
        expected.u1(theta, qr)
        expected.u1(phi, qr)
        expected.u1(sum_, qr)
        expected.u1(product_, qr)
        expected.u1(0.5, qr)

        after = Optimize1qGates().run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


class TestOptimize1qGatesParamReduction(QiskitTestCase):
    """Test for 1q gate optimizations parameter reduction, reduce n in Un """

    def test_optimize_u3_to_u2(self):
        """U3(pi/2, pi/3, pi/4) ->  U2(pi/3, pi/4)"""
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u3(np.pi / 2, np.pi / 3, np.pi / 4, qr[0])

        expected = QuantumCircuit(qr)
        expected.u2(np.pi / 3, np.pi / 4, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates())
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_to_u2_round(self):
        """U3(1.5707963267948961, 1.0471975511965971, 0.7853981633974489) ->  U2(pi/3, pi/4)"""
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u3(1.5707963267948961, 1.0471975511965971, 0.7853981633974489, qr[0])

        expected = QuantumCircuit(qr)
        expected.u2(np.pi / 3, np.pi / 4, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates())
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_to_u1(self):
        """U3(0, 0, pi/4) ->  U1(pi/4)"""
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u3(0, 0, np.pi / 4, qr[0])

        expected = QuantumCircuit(qr)
        expected.u1(np.pi / 4, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates())
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_to_u1_round(self):
        """U3(1e-16, 1e-16, pi/4) ->  U1(pi/4)"""
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u3(1e-16, 1e-16, np.pi / 4, qr[0])

        expected = QuantumCircuit(qr)
        expected.u1(np.pi / 4, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates())
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)


class TestOptimize1qGatesBasis(QiskitTestCase):
    """Test for 1q gate optimizations parameter reduction with basis """

    def test_optimize_u3_basis_u3(self):
        """U3(pi/2, pi/3, pi/4) (basis[u3]) ->  U3(pi/2, pi/3, pi/4)"""
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u3(np.pi / 2, np.pi / 3, np.pi / 4, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(['u3']))
        result = passmanager.run(circuit)

        self.assertEqual(circuit, result)

    def test_optimize_u3_basis_u2(self):
        """U3(pi/2, 0, pi/4) ->  U2(0, pi/4)"""
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u3(np.pi / 2, 0, np.pi / 4, qr[0])

        expected = QuantumCircuit(qr)
        expected.u2(0, np.pi / 4, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(['u2']))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_basis_u2_cx(self):
        """U3(pi/2, 0, pi/4) ->  U2(0, pi/4). Basis [u2, cx]."""
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u3(np.pi / 2, 0, np.pi / 4, qr[0])
        circuit.cx(qr[0], qr[1])

        expected = QuantumCircuit(qr)
        expected.u2(0, np.pi / 4, qr[0])
        expected.cx(qr[0], qr[1])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(['u2', 'cx']))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u1_basis_u2_u3(self):
        """U1(pi/4) ->  U3(0, 0, pi/4). Basis [u2, u3]."""
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u1(np.pi / 4, qr[0])

        expected = QuantumCircuit(qr)
        expected.u3(0, 0, np.pi / 4, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(['u2', 'u3']))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u1_basis_u2(self):
        """U1(pi/4) ->  Raises. Basis [u2]"""
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u1(np.pi / 4, qr[0])

        expected = QuantumCircuit(qr)
        expected.u3(0, 0, np.pi / 4, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(['u2']))
        with self.assertRaises(TranspilerError):
            _ = passmanager.run(circuit)

    def test_optimize_u3_basis_u2_u1(self):
        """U3(pi/2, 0, pi/4) ->  U2(0, pi/4). Basis [u2, u1]."""
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u3(np.pi / 2, 0, np.pi / 4, qr[0])

        expected = QuantumCircuit(qr)
        expected.u2(0, np.pi / 4, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(['u2', 'u1']))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_basis_u1(self):
        """U3(0, 0, pi/4) ->  U1(pi/4). Basis [u1]."""
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.u3(0, 0, np.pi / 4, qr[0])

        expected = QuantumCircuit(qr)
        expected.u1(np.pi / 4, qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(['u1']))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
