# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for pass cancelling 2 consecutive CNOTs on the same qubits."""

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import TemplateOptimization
from qiskit.test import QiskitTestCase


class TestTemplateMatching(QiskitTestCase):
    """Test the CXCancellation pass."""

    def test_pass_cx_cancellation(self):
        """Test the cx cancellation pass.

        It should cancel consecutive cx pairs on same qubits.
        """
        qr = QuantumRegister(2)
        circuit_in = QuantumCircuit(qr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[1], qr[0])
        circuit_in.cx(qr[1], qr[0])

        pass_manager = PassManager()
        pass_manager.append(TemplateOptimization())
        circuit_in_opt = pass_manager.run(circuit_in)

        circuit_out = QuantumCircuit(qr)
        circuit_out.h(qr[0])
        circuit_out.h(qr[0])

        self.assertEqual(circuit_in_opt, circuit_out)


'''    def test_optimize_h_gates_pass_manager(self):
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

        self.assertEqual(expected, result)'''
